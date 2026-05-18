//! Minimal scoped thread pool tuned for short-task parallel-for.
//!
//! Phase 8.A: redesigned around an atomic task counter. The previous design
//! (Phase 7.L) used one mpsc channel guarded by a mutex; workers contended on
//! `recv` for every task, and we paid one `Box<dyn FnOnce>` allocation per
//! task. With ~3,840 dispatches per `tg32` decode that cost dominated the
//! 1.4× gap vs llama.cpp (see Phase 7.U gap-analysis blog).
//!
//! Design:
//! - Workers pre-spawn once and block on a condvar waiting for a generation
//!   counter to advance.
//! - `parallel_for` installs `(f, n_tasks)` into a shared slot, resets the
//!   atomic counter, bumps the generation, and notifies workers.
//! - Each worker wakes once and loops `fetch_add(next_task)` until the counter
//!   meets `n_tasks`. No per-task mutex; no per-task allocation.
//! - Caller blocks on a completion condvar; workers signal when the last one
//!   finishes.
//!
//! Tradeoffs:
//! - One `unsafe` cast extends the closure borrow's lifetime for the duration
//!   of the call. SAFETY rests on the caller blocking until every worker has
//!   acknowledged completion before returning.
//! - `Drop` is not implemented: the only instance is the process-wide
//!   singleton (`global()`) and the OS reaps worker threads at exit.

use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::{self, JoinHandle};

/// Type-erased pointer to a `&(dyn Fn(usize) + Send + Sync)` valid for the
/// duration of a `parallel_for` call.
type FnPtr = *mut ();

/// State shared between the caller and every worker. All fields are touched
/// per `parallel_for` call; ordering documented inline.
struct PoolState {
    /// Erased pointer to the user closure; published with Release in
    /// `parallel_for`, observed with Acquire in workers.
    job_fn: AtomicPtr<()>,
    /// Total tasks for the current job.
    n_tasks: AtomicUsize,
    /// Cursor workers `fetch_add` into to claim tasks.
    next_task: AtomicUsize,

    /// Bumps once per `parallel_for`. Workers compare against their local
    /// `last_gen` to detect a new job without re-reading `job_fn`.
    job_generation: AtomicUsize,
    /// Workers wait on this when idle; caller notifies after publishing a job.
    job_mutex: Mutex<()>,
    job_cv: Condvar,

    /// Incremented by each worker exactly once per job after it drains tasks.
    finished_workers: AtomicUsize,
    /// Caller waits on this for `finished_workers == n_workers`.
    completion_mutex: Mutex<()>,
    completion_cv: Condvar,

    /// How many workers belong to this pool; cached here so workers can compute
    /// "am I the last one?" without crossing the pool boundary.
    n_workers: usize,
}

/// A fixed-size pool of worker threads driven by an atomic task counter.
pub struct ThreadPool {
    state: Arc<PoolState>,
    n_workers: usize,
    #[allow(dead_code)]
    workers: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    pub fn new(n_workers: usize) -> Self {
        assert!(n_workers >= 1, "ThreadPool needs at least 1 worker");
        let state = Arc::new(PoolState {
            job_fn: AtomicPtr::new(std::ptr::null_mut()),
            n_tasks: AtomicUsize::new(0),
            next_task: AtomicUsize::new(0),
            job_generation: AtomicUsize::new(0),
            job_mutex: Mutex::new(()),
            job_cv: Condvar::new(),
            finished_workers: AtomicUsize::new(0),
            completion_mutex: Mutex::new(()),
            completion_cv: Condvar::new(),
            n_workers,
        });
        let workers = (0..n_workers)
            .map(|idx| {
                let state = state.clone();
                thread::Builder::new()
                    .name(format!("lumen-pool-{}", idx))
                    .spawn(move || worker_loop(state))
                    .expect("spawn pool worker")
            })
            .collect();
        Self {
            state,
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

        // SAFETY: the pointer below is dereferenced only by workers, which
        // we block on (`completion_cv`) before this function returns. The
        // borrow `&f` therefore outlives every use of the raw pointer.
        let f_ref: &(dyn Fn(usize) + Send + Sync) = &f;
        let fat_ptr: *const (dyn Fn(usize) + Send + Sync) = f_ref;
        let fn_box: Box<*const (dyn Fn(usize) + Send + Sync)> = Box::new(fat_ptr);
        let fn_ptr = Box::into_raw(fn_box) as *mut ();

        // Reset per-job counters and publish the job.
        self.state.next_task.store(0, Ordering::Relaxed);
        self.state.finished_workers.store(0, Ordering::Relaxed);
        self.state.n_tasks.store(n_tasks, Ordering::Relaxed);
        self.state.job_fn.store(fn_ptr, Ordering::Release);

        // Bump generation and wake workers under the mutex so wakeups can't
        // race with workers re-checking `job_generation` and going back to
        // sleep.
        {
            let _g = self.state.job_mutex.lock().expect("job mutex");
            self.state.job_generation.fetch_add(1, Ordering::Release);
            self.state.job_cv.notify_all();
        }

        // Wait for every worker to acknowledge completion of this job.
        {
            let mut g = self.state.completion_mutex.lock().expect("completion mutex");
            while self.state.finished_workers.load(Ordering::Acquire) < self.n_workers {
                g = self
                    .state
                    .completion_cv
                    .wait(g)
                    .expect("completion condvar");
            }
        }

        // Reclaim the heap box. SAFETY: every worker stopped dereferencing
        // the pointer before bumping `finished_workers`, and we already
        // observed `finished_workers == n_workers` above.
        unsafe {
            drop(Box::from_raw(
                fn_ptr as *mut *const (dyn Fn(usize) + Send + Sync),
            ));
        }
    }
}

fn worker_loop(state: Arc<PoolState>) {
    let mut last_gen: usize = 0;
    loop {
        // Wait for a new job. The generation counter bumps once per
        // parallel_for; comparing against `last_gen` avoids missed wakeups.
        {
            let mut guard = state.job_mutex.lock().expect("worker job mutex");
            while state.job_generation.load(Ordering::Acquire) == last_gen {
                guard = state.job_cv.wait(guard).expect("worker job condvar");
            }
            last_gen = state.job_generation.load(Ordering::Acquire);
        }

        // Read job. The Release store in parallel_for happens-before this
        // Acquire load via the condvar handoff, so the pointer is valid.
        let fn_ptr = state.job_fn.load(Ordering::Acquire) as *const *const (dyn Fn(usize) + Send + Sync);
        let n_tasks = state.n_tasks.load(Ordering::Relaxed);
        // SAFETY: see parallel_for; the box stays alive until every worker
        // has bumped `finished_workers`.
        let f: &(dyn Fn(usize) + Send + Sync) = unsafe { &**fn_ptr };

        // Drain tasks from the shared counter. fetch_add gives each worker
        // a contiguous-but-interleaved run; for our typical 8-worker /
        // 8-task pattern this hands every worker exactly one task with no
        // contention beyond the single atomic increment.
        loop {
            let i = state.next_task.fetch_add(1, Ordering::Relaxed);
            if i >= n_tasks {
                break;
            }
            f(i);
        }

        // Signal completion. The last worker wakes the caller.
        let prev = state.finished_workers.fetch_add(1, Ordering::AcqRel);
        if prev + 1 == state.n_workers {
            let _g = state.completion_mutex.lock().expect("worker completion mutex");
            state.completion_cv.notify_one();
        }
    }
}

static GLOBAL_POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Lazily-initialized process-wide pool. Worker count is the system's
/// available parallelism, capped at 8 to bound L3 contention on bigger boxes.
///
/// Override via `LUMEN_THREADS=N` env var (Phase 7.U: scaling experiments).
pub fn global() -> &'static ThreadPool {
    GLOBAL_POOL.get_or_init(|| {
        let n = std::env::var("LUMEN_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
                    .min(8)
            });
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
