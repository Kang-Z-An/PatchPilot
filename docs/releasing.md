# Preparing a public release

Run from a clean checkout of the commit intended for publication:

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python examples/offline_demo.py
uv run python scripts/check_release.py
uv build
git diff --check
```

The release checker reads publishable working-tree files, every commit reachable
from local refs, and commit/tag metadata. It reports paths and rule names, never
matched secret values. It checks common credential patterns, exact key values
from the ignored local `.env`, local home paths, and private artifact paths.
Only documented synthetic fixtures/reserved example hosts are exempted. This is
a heuristic scan; review the intended diff and packaged files as well.

Inspect the wheel and source archive under `dist/` before distributing them.
The wheel contains the PatchPilot package and its license. The source archive
also contains documentation, tests, benchmark records, and third-party notices.
Neither should contain `.env`, `.patchpilot/`, private credentials, raw model
conversations, weights, or third-party checkouts. Git commits normally retain
author names/emails; adding a file to `.gitignore` does not remove its history.

Record official benchmark outcomes separately from Agent verification and
training-data acceptance. Existing experiment run IDs such as `v056-...` are
experiment labels, not Git release tags. Do not reuse an existing version tag
for a different commit or claim GitHub CI passed before its run completes.

Before publishing, choose the repository owner/name, authenticate GitHub CLI
using `gh auth login`, inspect `git remote -v`, and commit the intended changes.
Create or connect the selected repository, push the chosen branch, and check
the first CI run. Configure private vulnerability reporting for the public
repository when available. Package upload to PyPI is a separate release action;
the local project name does not establish PyPI name ownership.
