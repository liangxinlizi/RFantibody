# NPU-aware original test surface (implementation record)

Status: **implementation only — no probe, build, install, pytest, E2E, parity or
profiler command was executed while producing this file.** The `runExperiments`
operation that wrote this change is not allowed to execute project commands, so
the measurements required by the iteration-14 proposal (§6.1 diagnostics D0/W/G,
the before/after junit counts, the per-case device evidence and the `.venv`
immutability hash) are **not yet collected**. They must be recorded here by the
deterministic evaluation phase. Nothing in this document is a measured result.

## 1. Change

| Path | Kind | Change |
|---|---|---|
| `test/util/device_selection.py` | new | `npu_device_name()`, `cuda_device_name()`, `reference_set_dir()` — genuine accelerator detection and reference-set selection. No assertions, no tolerances, no reference data. |
| `test/rfdiffusion/conftest.py` | edit | `check_gpu` gains an NPU branch **before** the unchanged CUDA requirement; `ref_dir` delegates to `reference_set_dir()`. |
| `test/proteinmpnn/conftest.py` | edit | same edit as above. |
| `test/rf2/conftest.py` | edit | same edit as above. |
| `test/rf2/scripts/json_prediction.sh` | edit | weights directory resolved as `"${RFANTIBODY_WEIGHTS:-$PROJECT_DIR/weights}"`; model, `--num-recycles 10`, `--seed 42`, input JSON, output naming and the determinism exports unchanged. |
| `README.md` | edit | documents the git-ignored `weights` symlink / `RFANTIBODY_WEIGHTS` default location and the NPU reference-set selection. |
| `weights` (git-ignored symlink) | environment | points at the externally supplied read-only checkpoints; no checkpoint is copied or tracked. |

Untouched, deliberately: every test module (`test_*/test_*.py`), every
assertion, tolerance, marker and comparison threshold, all reference assets,
`test/conftest.py`, `test/quiver/**`, `test/util/{test_util.py,util_test_utils.py,conftest.py}`,
`test/graph_parity/**`, and every product file (`src/`, `scripts/`, `include/`,
`csrc/`, `e2e_run.py`), `pyproject.toml`, `uv.lock`, `requirements-cuda.txt`,
`migration.yaml`, `task/tools/*`, `.venv/**`.

## 2. Why (root cause addressed)

The three module `conftest.py` files gated their suites on
`torch.cuda.is_available()` plus an `A4000`/`H100` device name, so on this
single-`Ascend910B2C` host all 14 module cases were skipped with
`No GPU found, tests require a supported GPU (A4000 or H100)` before any work
ran, and `ref_dir` fell back to the empty `reference_outputs/` directory. The
NPU is genuine (`torch.npu.is_available()` / `device_count()` /
`get_device_name()` from the inherited `torch_npu 2.5.1`), so the gate is
extended rather than spoofed: on a genuine NPU the suites execute and select the
versioned `H100_references` set, which is the only GPU reference set versioned
for ProteinMPNN and RF2 and the same numerical reference the CUDA path uses on
an H100. The CUDA branch is preserved verbatim, including the "supported GPU"
requirement and the A4000/H100 selection.

`test/rf2/scripts/json_prediction.sh` hardcoded
`$PROJECT_DIR/weights/RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt`,
bypassing the repository's documented `PathConfig.WEIGHTS_DIR`
(`RFANTIBODY_WEIGHTS` → `<project>/weights`). It now follows that documented
configuration, so externally supplied weights work without copying checkpoints
into the tree. The repository default location is provided as a git-ignored
symlink (`git check-ignore -v weights` → `.gitignore:6:weights`).

## 3. Measurements to be recorded by evaluation (exact commands)

Run from `workspace/project`. Verbatim output belongs in this file, replacing
this section.

```bash
# D0: the accelerator the tests will see, with no spoofing.
.venv/bin/python - <<'PY'
import torch
try:
    import torch_npu
    npu = bool(torch.npu.is_available()), int(torch.npu.device_count()), str(torch.npu.get_device_name(0))
except Exception as exc:
    npu = ("torch_npu import/device error", type(exc).__name__, str(exc))
print("cuda_available=", torch.cuda.is_available())
print("npu=", npu)
PY

# W: no-copy weight wiring, and proof it stays out of Git.
readlink -f weights
for f in RFdiffusion_Ab.pt ProteinMPNN_v48_noise_0.2.pt RF2_ab.pt RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt; do
  test -s "weights/$f" && sha256sum "weights/$f"
done
git check-ignore -v weights && git status --porcelain

# G: negative control retained for the record — with the NPU absent the original
# message is unchanged; on this host the cases must now execute.
git rev-parse HEAD
.venv/bin/python -m pip list --format=freeze | sha256sum     # before

# Per-module runs, before the full suite, so failures are attributable.
.venv/bin/python -m pytest test/rfdiffusion -ra
.venv/bin/python -m pytest test/proteinmpnn -ra
.venv/bin/python -m pytest test/rf2 -ra

# Full suite (project.test_command), in the plain evaluator environment
# (no injected RFANTIBODY_WEIGHTS / RFANTIBODY_ROOT).
.venv/bin/python -m pytest test -ra --junitxml=../../migration_reports/full-tests.xml

.venv/bin/python -m pip list --format=freeze | sha256sum     # after (must match)
git status --porcelain                                       # ignored artifacts only
```

Record in this file, from those logs and the junit XML:

* D0 device tuple (`cuda_available=False`, genuine `Ascend910B2C` name) and the
  canonical external weights path with the four SHA-256 digests, the ignore rule
  and the clean `git status`.
* `HEAD` commit and the statement that only the six §1 paths differ.
* junit `collected` / `passed` / `failed` / `skipped` before and after, with the
  before-counts taken from `ae-temp/latest-run/full-tests.xml` (iteration 13:
  101 / 87 / 0 / 14).
* The zero-skip proof: an empty skip-id list.
* Per-case outcome for the 14 previously skipped ids — the five CDR/indexing
  cases and the nine script/reference-comparison cases — with the failing
  comparison's measured differences (first messages, residue/atom counts, max
  coordinate deviations where present) and per-case device evidence (the device
  name printed by `check_gpu`, the stage CLI's accelerator line, absence of any
  CPU-fallback marker).
* `.venv` pip-freeze hash before/after and the `git status` output.

## 4. Statements that must accompany the measurements

* No assertion, tolerance, marker, test command, reference asset or comparison
  threshold was changed by this iteration; the references are the versioned
  `H100_references` sets, never generated from candidate output.
* Remaining reference-comparison mismatches, if any, are the measured
  CUDA-vs-NPU stochastic-stream fingerprint (O7c). They are recorded as
  failures, never claimed as passes.
* `.venv` content and dependency metadata are unchanged by this iteration, and
  `git status` is clean apart from git-ignored artifacts.
* The suite is executed with the plain evaluator environment (no injected
  `RFANTIBODY_WEIGHTS`), so the documented default resolution is what the
  evaluator exercises.

## 5. Out of scope here

Product RNG equivalence (O7c: Philox normal/Bernoulli/multinomial stream
compatibility, `torch_npu` RNG handling, decoding order, RF2 stochasticity), the
acceptance chain (O8), any product/kernel/graph change, any dependency or
`.venv` change, any test weakening, any device spoof or fake `uv` wrapper, and
any copying of checkpoints into the tree.
