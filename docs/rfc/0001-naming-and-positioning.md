# RFC 0001 — Naming and Positioning

- Status: Draft
- Authors: redchupa
- Created: 2026-05-15
- Decision deadline: before v0.1.0 release

## Summary

Decide the project's canonical name and one-line positioning. Current working
name `Lumen` is generic and collides with many existing GitHub repos.

## Why this matters

1. **Discoverability**: a SEO-collision-prone name buries the repo behind 50
   unrelated projects.
2. **Crate name**: `cargo publish` needs a name that isn't already taken on
   crates.io.
3. **Positioning sentence**: the README's first line decides whether a stranger
   keeps reading. "Korean LLM runtime" is weak; "IR that auto-synthesizes
   quantized kernels" is the actual differentiator.

## Candidate names

| Name | Origin | Pros | Cons |
|---|---|---|---|
| `lumen` (current) | Latin "light" | Easy to read | crates.io likely taken; SEO buried |
| `iro` | 한국어 "색" | Short, distinct | Sounds like Spanish "go" |
| `sori` | 한국어 "소리" | Distinct, easy | Less obvious meaning to non-Koreans |
| `karak` | 한국어 "가락" | Unique | Two syllables |
| `pum` | 한국어 "품" | Short, evocative ("style/grace") | Phonetics in English |
| `lustr` | Latin-derived | Short, unused on crates.io (check) | Looks like a typo |

## Positioning sentence options

A. "A self-hosted LLM inference compiler that auto-synthesizes quantized kernels."
B. "Korean-first LLM runtime built on a custom DSL + IR + JIT."
C. "Write tensor code once. Lumen compiles it for every quant format × CPU arch."

A is currently in README. B is what was originally drafted. C is shortest.

## Decision criteria

1. crates.io availability (must)
2. GitHub search rank (top 10 in first page for the bare term)
3. Pronounceable in English and Korean
4. < 6 letters preferred (CLI typing)

## Action items

- [ ] Check crates.io for each candidate
- [ ] Check `gh search repos <name>` ranking
- [ ] Lock name by Phase 2 end (when first code generation lands)
