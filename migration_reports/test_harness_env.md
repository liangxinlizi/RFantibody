# Test-harness installed-environment binding (implementation record)

Status: **implementation only — no probe, build, install, pytest, E2E or profiler
command was executed while producing this file.** The `runExperiments` operation
that wrote this change is not allowed to execute project commands, so the
measurements required by the iteration-13 proposal (§6.1 diagnostic D1–D4, the
before/after full-suite counts, the `.venv`-immutability hash and the
`git status --porcelain` check) are **not yet collected**. They must be recorded
here by the deterministic evaluation phase. Nothing in this document is a
measured result.

## 1. Change

Single source change: `test/conftest.py`.

* Added `PROJECT_ROOT`, `_installed_venv()`, `_bind_installed_environment()`,
  a `pytest_configure` hook and a session-scoped autouse fixture
  `installed_project_environment`.
* `pytest_addoption` and its `--keep-outputs` option are unchanged.
* No test module, test command, assertion, marker, tolerance, reference asset,
  product file, dependency metadata or file under `.venv/` was modified.

`_installed_venv()` resolves the environment from the running interpreter
(`Path(sys.executable).resolve().parent.parent`, i.e. `.venv` on the NPU host or
`.venv-cuda` on the documented CUDA path) and falls back to
`PROJECT_ROOT/.venv`; both candidates must contain `pyvenv.cfg` and
`bin/python`. `_bind_installed_environment()` then applies, before collection:

| Variable | Value |
|---|---|
| `PATH` | `<venv>/bin` prepended (de-duplicated, nothing removed) |
| `VIRTUAL_ENV` | `<venv>` |
| `UV_PROJECT_ENVIRONMENT` | `<venv>` |
| `UV_NO_SYNC` | `1` |

This is the activation-equivalent binding plus the two uv environment pins that
the host CUDA reference used while passing the same original tests
(`migration_reports/host-cuda-environment-review.md`), and the same recipe that
`README.md` documents for `uv run` from arbitrary working directories. The real
`/usr/local/bin/uv` binary and the real installed console scripts are used; no
wrapper executable, no PATH shadowing and no test modification are introduced.
`<venv>/bin` contains no `uv`, `cmake`, `ccec` or `msprof`, so no tool is
shadowed.

## 2. Why (root cause addressed)

The run gate fails on exactly 9 unchanged original tests:

* `test/quiver/test_quiver.py::TestQuiverCLI::*` (7 cases) run
  `uv run qv* …` with `cwd=/tmp/quiver_test_*` through
  `subprocess.run(..., shell=True)` without `env=`. The pytest process is
  `.venv/bin/python -m pytest` and is *not* activated, so from `/tmp` uv finds no
  project and no active environment and cannot spawn the console script
  (`Failed to spawn: 'qvls' … No such file or directory (os error 2)`).
* `test/util/scripts/{antibody,nanobody}_hlt_conversion.sh` call bare `python`;
  without the venv first on `PATH` that is the system interpreter, which lacks
  `biotite` (`ModuleNotFoundError: No module named 'biotite'`).

The installed environment itself is complete (console scripts `qvfrompdbs` …
`qvscorefile`, `rfdiffusion`, `proteinmpnn`, `rf2`, `python`, `python3`, and
`biotite 1.2.0`, `dgl 1.1.3`, `pytest 9.1.1`); the children simply could not see
it.

## 3. Measurements to be recorded by evaluation (exact commands)

Run from `workspace/project`; `PROJECT=$PWD`, `VENV=$PROJECT/.venv`,
`TMP=$(mktemp -d)`:

```bash
# D1 negative control: must reproduce the baseline spawn error.
( cd "$TMP" && uv run qvls "$PROJECT/test/quiver/inputs_for_test/designs_scored.qv" ) ; echo "D1_exit=$?"

# D2 binding: must list all 10 scored tags, exit 0.
( cd "$TMP" && env PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV" \
    UV_PROJECT_ENVIRONMENT="$VENV" UV_NO_SYNC=1 \
    uv run qvls "$PROJECT/test/quiver/inputs_for_test/designs_scored.qv" ) ; echo "D2_exit=$?"

# D3 binding, write path: must create test.qv.
( cd "$TMP" && env PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV" \
    UV_PROJECT_ENVIRONMENT="$VENV" UV_NO_SYNC=1 \
    uv run qvfrompdbs "$PROJECT"/test/quiver/inputs_for_test/*.pdb > test.qv ) ; echo "D3_exit=$?"
grep -c '^QV_TAG' "$TMP/test.qv"

# D4 bare-python util script under the same binding.
mkdir -p "$TMP/util"
( cd "$TMP" && env PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV" \
    bash "$PROJECT/test/util/scripts/antibody_hlt_conversion.sh" "$TMP/util" ) ; echo "D4_exit=$?"
ls -l "$TMP/util/antibody_HLT.pdb"
```

Also record: `uv --version`, `.venv/pyvenv.cfg`, the effective `PATH` head,
`pip list --format=freeze | sha256sum` before and after the suite (must be
identical), `git status --porcelain` after the suite, and the before (iteration
12 junit `ae-temp/latest-run/full-tests.xml`) and after full-suite counts plus
passed/failed/skipped id sets.

## 4. Interpretation and open hypotheses

* Expected transition of the `run` gate: from `9 failed / 78 passed /
  14 skipped` (exit 1) to `0 failed / 87 passed / 14 skipped` (exit 0), with the
  7 Quiver and 2 util cases executed and passed, and the junit skip id set
  exactly the 14 baseline CUDA-only ids (`rfdiffusion` 4, `proteinmpnn` 7,
  `rf2` 3). Those 14 skips are **not** addressed here; they remain the open O7b
  item and must stay labelled as such.
* D1 must fail with the `Failed to spawn` message. If D1 passes, the environment
  is already bound by something else and the diagnosis must be re-recorded before
  relying on this change.
* **Unresolved hypothesis (not measurable in this operation):** whether uv
  `0.12.0` resolves `uv run <console-script>` from a directory with no
  `pyproject.toml` using only `PATH` + `VIRTUAL_ENV` +
  `UV_PROJECT_ENVIRONMENT` + `UV_NO_SYNC`. The host CUDA recipe additionally
  exported `UV_PROJECT=<repo>`, so if D2/D3 fails with a project-resolution or
  lockfile message, the bounded fallback from the proposal is to add
  `os.environ.setdefault("UV_PROJECT", str(PROJECT_ROOT))` to
  `_bind_installed_environment()` — no `--frozen`/`--locked`, no `uv.lock` edit,
  no wrapper, no change to any test. Otherwise `UV_PROJECT` is intentionally not
  set.
* If a Quiver or util case still fails after the binding with `Failed to spawn`
  or `No module named 'biotite'`, the mechanism is wrong (implementation
  failure), not progress. A failure on a genuine product assertion is a new
  fingerprint to report verbatim.
* Non-regression to verify: build exit 0 with eight `aclrtlaunch_*` entries and
  artifact sha256 `e573408f…`, install exit 0, graph parity 78/78, accuracy gate
  unchanged, NPU probe gate unchanged, `project.target_command` exit 0 with the
  three NPU stages. This change touches only pytest process environment plumbing,
  so product behaviour must be identical.
* Known side effect to record rather than commit: `qvscorefile` rewrites
  `test/quiver/inputs_for_test/designs_scored.sc` next to its input; if
  `git status --porcelain` shows it modified, it is a harness side effect of the
  unchanged original test and must not be committed.
