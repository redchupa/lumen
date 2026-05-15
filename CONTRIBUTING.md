# Contributing to Lumen

Lumen is being built in public, Phase by Phase. Before contributing:

1. Read [PLAN.md](./PLAN.md) — which Phase your change belongs to.
2. Read [CLAUDE.md](./CLAUDE.md) — coding rules, especially the dependency policy.
3. Read [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md).

## Development setup

```sh
# Install Rust 1.78+
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# or on Windows: winget install Rustlang.Rustup

git clone https://github.com/redchupa/lumen
cd lumen
cargo build --workspace
cargo test --workspace
```

## Pull request checklist

- [ ] `cargo fmt --all` clean
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean
- [ ] `cargo test --workspace` passes
- [ ] Public items have rustdoc
- [ ] New ops/passes have a correctness test
- [ ] Performance-affecting changes have a criterion bench

## Commit style

```
[lumen-ir] add SSA Value type

Optional body explaining the why. Reference Phase if relevant.

Co-Authored-By: Claude <noreply@anthropic.com>
```

## RFC process

Non-trivial design changes (new IR op, new backend, new file format) go through
RFC first: open a draft PR adding `docs/rfc/NNN-title.md`. Discuss for ≥ 3 days
before merging implementation.

## Code of conduct

Be nice. Hostile/personal comments will be removed.
