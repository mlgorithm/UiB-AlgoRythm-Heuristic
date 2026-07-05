# UiB-AlgoRythm — PACE 2026 Heuristic Track (`algo`)

Single-file C++17 solver for the [PACE 2026](https://pacechallenge.org/2026/) rooted
**Maximum Agreement Forest (MAF)** problem on two phylogenetic trees. It minimizes the number
of agreement-forest components `k` (equivalently the rooted SPR distance `k − 1`).

See [`SOLVER_DESCRIPTION.md`](SOLVER_DESCRIPTION.md) for the algorithm.

## Installation

**External dependencies: none.** A C++17 compiler (GCC ≥ 9 or Clang ≥ 10) is the only requirement.

```bash
g++ -O3 -std=c++17 -pipe -o pace2026_heuristic main.cpp
```

A Debian 13.5 container setup is provided in [`docker_setup.sh`](docker_setup.sh).

## Run

```bash
PACE_TIME_LIMIT=298 ./pace2026_heuristic < instance.nw > solution.txt
```

Reads one PACE `.nw` instance on stdin and writes the agreement forest to stdout — one Newick
component per line, `;`-terminated, singletons as bare integers. Anytime and SIGTERM-safe (on
SIGTERM it flushes the best valid forest seen so far). Single-threaded, ≤8 GB.

Env toggles (default: both features on): `PACE_NOKERNEL=1`, `PACE_NOEXACT=1`, `PACE_KDEBUG=1`.

## Verifying a solution

`tests/maf_check.py` is a dependency-free, independent checker (it shares no code with the solver):

```bash
python3 - <<'PY'
import tests.maf_check as m
n,t,trees = m.parse_instance(open('instance.nw').read())
masks = m.output_to_masks(n, open('solution.txt').read())
ok, reason = m.validate_forest(n, trees, masks)
print('VALID' if ok else 'INVALID: '+reason, 'k=%d' % len(masks))
PY
```

## License

MIT — see [`LICENSE`](LICENSE).
