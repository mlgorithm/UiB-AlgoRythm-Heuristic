# PACE 2026 Heuristic Track — Solver Description (`algo`)

> **Contact (required on the final PDF title page):** `<your-email@example.com>` — replace before submitting.
>
> **Note:** PACE requires the solver description as a **PDF** (LIPIcs template recommended, ≤4 pages,
> with the contact email on the title page). This markdown is the source content; convert it to the
> required PDF before the description deadline.

## Problem

Rooted **Maximum Agreement Forest (MAF)** on two rooted binary phylogenetic trees
`T1`, `T2` over the same leaf set (size `n`). We output a partition of the leaves
into agreement-forest components — each component induces the same rooted topology
in both trees and the components are edge-disjoint in each tree — minimizing the
number of components `k` (equivalently the rooted SPR distance `k − 1`).

Single C++17 file, anytime, ≤8 GB, SIGTERM-safe. Reads one PACE `.nw` instance on
stdin, writes one Newick component per line to stdout. All computation runs on a
single worker thread (the main thread only blocks on it), so it is single-threaded
in the PACE sense — the worker just carries a large (1 GiB) stack so that a deeply
nested (caterpillar/ladder) input cannot overflow the default stack in any recursive
routine, which would otherwise SIGSEGV with empty output.

## Algorithm (pipeline)

1. **Singleton fallback published first.** Before any search, the trivial all-singleton
   forest is cached, so a SIGTERM at any moment flushes a valid forest.

2. **Lossless common-subtree kernelization.** Iteratively collapse every *common
   cherry* (two leaves that are siblings in **both** trees) into a single super-leaf,
   cascading to collapse maximal common pendant subtrees. This is the classic
   Bordewich–Semple subtree reduction: it is **distance/optimality preserving** (a
   common pendant subtree is intra-component in every MAF), so the reduced instance has
   the same optimum. We solve the smaller kernel and expand each super-leaf back to its
   subtree at output time. Shrinks structured mid/large instances by 20–35% and lets the
   throughput-bound search do more effective work.

3. **Merge-based construction + improvement.** A common-clade sweep builds an initial
   forest; a stack of improvement passes (merge / component-pack / corridor-merge /
   window repair / conflict repair) greedily combines compatible components.

4. **Never-regress exact-core repair.** Small clean common clades (a clade in both trees
   that is exactly a union of whole current components) are re-solved to **proven
   optimum** by an exact Whidden–Beiko–Zeh rooted-MAF branch-and-bound, and spliced back
   only when this strictly reduces the component count. The exact solver is node- and
   wall-time-capped; if it does not finish, the heuristic result is kept. This can never
   worsen the forest.

5. **Seed portfolio + anytime restart loop.** A portfolio of alternative constructors
   (MAST-based, greedy, cherry-cut beam, pair-portfolio) and a random-shatter
   large-neighborhood restart loop spend the remaining budget, keeping the best forest
   seen.

6. **Always-on feasibility gate.** Every forest is re-checked (partition + per-component
   both-tree topology agreement + edge-disjointness) before it can become the published
   best; an infeasible candidate is refused and the previous valid best is kept. The
   SIGTERM handler is async-signal-safe and flushes only validated output.

## Budget / interface

- Time budget from `PACE_TIME_LIMIT` seconds (default 298). On SIGTERM the cached best
  forest is flushed via `write(2)`.
- Optional env toggles (default: both on): `PACE_NOKERNEL=1` disables kernelization,
  `PACE_NOEXACT=1` disables exact-core repair, `PACE_KDEBUG=1` prints stats to stderr.

## Correctness / verification

- The exact B&B is fuzz-verified against a brute-force oracle (2000+ instances, n ≤ 11)
  and an independent Whidden implementation (n ≤ 15): 0 mismatches.
- Kernelization is optimality-preserving (verified: brute-force optimum of the kernel
  equals that of the original on thousands of small instances) and its expansion is a
  bijection on leaves.
- Every output is validated by an independent checker (`tests/maf_check.py`) across the
  size range including SIGTERM-interrupted runs on the largest instances.

## Citations (algorithms; all code is an original implementation, no third-party source copied)

- C. Whidden, R. G. Beiko, N. Zeh. *Fixed-Parameter Algorithms for Maximum Agreement
  Forests.* SIAM J. Comput. 42(4), 2013. (exact rooted-MAF branch-and-bound)
- M. Bordewich, C. Semple. *On the computational complexity of the rooted subtree prune
  and regraft distance.* Ann. Comb. 8, 2005. (subtree reduction)
