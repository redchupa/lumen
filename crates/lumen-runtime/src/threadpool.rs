//! Minimal scoped thread pool tuned for short-task parallel-for.
//!
//! Phase 7.L: replaces rayon for the decode-time Q8 matmul fan-out. rayon's
//! work-stealing machinery is overkill (and not free) for the pattern we
//! actually have — a small fixed number of equal-sized chunks, dispatched
//! many times per token. This pool pre-spawns N worker threads (once, lazily)
//! that pull from a single mpsc channel, and provides a `parallel_for`
//! barrier-style call.
//!
//! Tradeoffs we accepted:
//! - Single mpsc channel guarded by a mutex (so workers contend on `recv`).
//!   With ≤ 8 workers and tasks that take tens of microseconds each, this
//!   is fine. A per-worker queue with work-stealing would be faster on
//!   ragged workloads, but we don't have those.
//! - One `unsafe` transmute extends the user closure's lifetime to `'static`
//!   for the `Box<dyn FnOnce>` requirement. SAFETY rests on `parallel_for`
//!   not returning until every task has finished — enforced by the pending
//!   counter + condvar wait.
//! - `Drop` is not implemented for `ThreadPool` because the only instance is
//!   the global singleton, which lives until process exit; OS reaps the
//!   worker threads.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::{self, JoinHandle};

type Job = Box<dyn FnOnce() + Send + 'static>;

struct WaitState {
    pending: AtomicUsize,
    mutex: Mutex<()>,
    cv: Condvar,
}

/// A fixed-size pool of worker threads that consume jobs from a shared queue.
pub struct ThreadPool {
    tx: Sender<Job>,
    n_workers: usize,
    // Kept alive so threads aren't reaped prematurely; we never join them
    // ourselves (see the module note about Drop).
    #[allow(dead_code)]
    workers: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    pub fn new(n_workers: usize) -> Self {
        assert!(n_workers >= 1, "ThreadPool needs at least 1 worker");
        let (tx, rx) = mpsc::channel::<Job>();
        let rx: Arc<Mutex<Receiver<Job>>> = Arc::new(Mutex::new(rx));
        let workers = (0..n_workers)
            .map(|idx| {
                let rx = rx.clone();
                thread::Builder::new()
                    .name(format!("lumen-pool-{}", idx))
                    .spawn(move || worker_loop(rx))
                    .expect("spawn pool worker")
            })
            .collect();
        Self {
            tx,
            n_workers,
            workers,
        }
    }

    pub fn n_workers(&self) -> usize {
        self.n_workers
    }

    /// Run `f(0)..f(n_tasks - 1)` across the pool, blocking until every task
    /// has finished. `f` is shared by reference across all tasks; it must be
    /// `Sync` because multiple workers may invoke it concurrently.
    pub fn parallel_for<F>(&self, n_tasks: usize, f: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        if n_tasks == 0 {
            return;
        }
        let wait = Arc::new(WaitState {
            pending: AtomicUsize::new(n_tasks),
            mutex: Mutex::new(()),
            cv: Condvar::new(),
        });
        // SAFETY: every Box we send below holds a `&'static dyn Fn(usize)`
        // forged from a borrow of `f` by transmute. The borrow is valid for
        // the duration of `parallel_for`; we block on `wait.pending` reaching
        // zero before returning, so no task outlives `f`'s real lifetime.
        let f_borrow: &(dyn Fn(usize) + Send + Sync) = &f;
        let f_static: &'static (dyn Fn(usize) + Send + Sync) =
            unsafe { std::mem::transmute(f_borrow) };

        for i in 0..n_tasks {
            let wait = wait.clone();
            self.tx
                .send(Box::new(move || {
                    f_static(i);
                    if wait.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
                        let _guard = wait.mutex.lock().expect("wait mutex");
                        wait.cv.notify_all();
                    }
                }))
                .expect("pool channel closed unexpectedly");
        }

        // Block until every task has decremented the counter.
        let mut guard = wait.mutex.lock().expect("wait mutex");
        while wait.pending.load(Ordering::Acquire) > 0 {
            guard = wait.cv.wait(guard).expect("wait condvar");
        }
    }
}

fn worker_loop(rx: Arc<Mutex<Receiver<Job>>>) {
    loop {
        // Hold the lock just long enough to claim a job; the Job itself runs
        // outside the lock so the rest of the workers can claim theirs.
        let job = {
            let guard = rx.lock().expect("worker rx mutex");
            guard.recv()
        };
        match job {
            Ok(job) => job(),
            // Sender dropped — pool tearing down.
            Err(_) => break,
        }
    }
}

static GLOBAL_POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Lazily-initialized process-wide pool. Worker count is the system's
/// available parallelism, capped at 8 to bound L3 contention on bigger boxes.
pub fn global() -> &'static ThreadPool {
    GLOBAL_POOL.get_or_init(|| {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(8);
        ThreadPool::new(n)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_for_runs_every_task_exactly_once() {
        let pool = ThreadPool::new(4);
        let counters: Vec<AtomicUsize> = (0..50).map(|_| AtomicUsize::new(0)).collect();
        pool.parallel_for(counters.len(), |i| {
            counters[i].fetch_add(1, Ordering::Relaxed);
        });
        for (i, c) in counters.iter().enumerate() {
            assert_eq!(c.load(Ordering::Relaxed), 1, "task {} ran wrong count", i);
        }
    }

    #[test]
    fn parallel_for_zero_tasks_is_noop() {
        let pool = ThreadPool::new(2);
        pool.parallel_for(0, |_| panic!("should not run"));
    }

    #[test]
    fn parallel_for_can_run_back_to_back() {
        let pool = ThreadPool::new(3);
        for round in 0..5 {
            let sum = AtomicUsize::new(0);
            pool.parallel_for(20, |i| {
                sum.fetch_add(i, Ordering::Relaxed);
            });
            assert_eq!(
                sum.load(Ordering::Relaxed),
                (0..20).sum::<usize>(),
                "round {}",
                round
            );
        }
    }

    #[test]
    fn global_pool_returns_same_instance() {
        let a = global() as *const ThreadPool;
        let b = global() as *const ThreadPool;
        assert_eq!(a, b);
    }
}
