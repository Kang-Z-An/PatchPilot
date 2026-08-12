# Requests environment analysis

`psf__requests-5414` is retained as **officially unresolved**. This note
documents why its local result is also marked `environment_suspect`; it does
not replace or adjust the official score.

## Patch comparison

PatchPilot changed the hostname guard from:

```python
elif host.startswith(u'*'):
```

to:

```python
elif host.startswith(u'*') or host.startswith(u'.'):
```

The SWE-bench gold patch uses the functionally equivalent tuple form:

```python
elif host.startswith((u'*', u'.')):
```

Both versions reject a hostname beginning with `.` by raising `InvalidURL`.

## Three-run evidence

The official harness was run three times under the same local amd64 Docker
environment:

| Run | Patch | FAIL_TO_PASS | PASS_TO_PASS | Official result |
|---|---|---:|---:|---|
| Initial batch | PatchPilot | 1/1 passed | 126 passed, 4 failed | unresolved |
| Single-instance retry | PatchPilot | 1/1 passed | 126 passed, 4 failed | unresolved |
| Gold control | SWE-bench gold | 1/1 passed | 126 passed, 4 failed | unresolved |

All three runs failed the same four regression tests:

- `tests/test_requests.py::TestTimeout::test_connect_timeout[timeout0]`
- `tests/test_requests.py::TestTimeout::test_connect_timeout[timeout1]`
- `tests/test_requests.py::TestTimeout::test_total_timeout_connect[timeout0]`
- `tests/test_requests.py::TestTimeout::test_total_timeout_connect[timeout1]`

The issue-specific `FAIL_TO_PASS` test succeeded in every run. Because the gold
patch produced exactly the same unrelated timeout failures, the evidence does
not attribute this unresolved result to PatchPilot's code change. Likely causes
include an incompatibility or service/timing issue in the local harness
environment.

## Reporting policy

The experiment headline remains **7/8 resolved (87.5%)**. We additionally
report one environment-suspect instance and 8/8 successful target tests. We do
not report an adjusted 100% SWE-bench score, because the official resolver
requires both `FAIL_TO_PASS` and `PASS_TO_PASS` to succeed.
