"""Read-only, redacted heuristic scan of publishable files and Git history."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from dotenv import dotenv_values


PATTERNS = {
    "api_key": re.compile(rb"\b(?:sk-|as_sk_)[A-Za-z0-9_-]{16,}\b"),
    "github_token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "hf_token": re.compile(rb"\bhf_[A-Za-z0-9]{25,}\b"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential_url": re.compile(rb"https?://[A-Za-z0-9._~%+-]+:[A-Za-z0-9._~%!$&()*+,;=:+-]+@"),
}
PRIVATE_DIRS = {".patchpilot", ".venv", "logs", "checkpoints", "wandb", "__pycache__"}
MAX_BYTES = 8 * 1024 * 1024


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def forbidden(path: str) -> bool:
    p = Path(path)
    return (
        any(part in PRIVATE_DIRS for part in p.parts)
        or (p.name.startswith(".env") and p.name != ".env.example")
        or p.suffix in {".pem", ".key", ".safetensors", ".pt", ".pth"}
        or path.startswith("benchmarks/results/")
    )


def scan_bytes(path: str, data: bytes, known_secrets: list[bytes]) -> set[str]:
    findings = set()
    for name, pattern in PATTERNS.items():
        for match in pattern.finditer(data):
            if name == "credential_url":
                host = re.match(rb"[A-Za-z0-9.-]+", data[match.end():])
                # Upstream URL-validation fixtures use reserved example domains.
                if host and host.group().lower() in {b"example", b"example.com", b"example.org", b"example.net"}:
                    continue
            # This exact synthetic value exercises the training-data rejection test.
            if path == "tests/test_training_data.py" and match.group() == b"sk-" + b"this-is-a-secret-value":
                continue
            findings.add(name)
    if any(value in data for value in known_secrets):
        findings.add("known_local_secret")
    if str(Path.home()).encode() + b"/" in data:
        findings.add("local_home_path")
    return findings


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    # Values are used only for exact matching; neither values nor matched lines
    # are emitted. Ignored local files themselves are not publication candidates.
    known = [
        value.encode() for name, value in dotenv_values(root / ".env").items()
        if value and len(value) >= 12 and re.search(r"KEY|TOKEN|SECRET", name)
        and not value.startswith(("replace-", "your-"))
    ]
    findings: list[dict[str, str]] = []
    scanned: set[tuple[str, str]] = set()

    def inspect(path: str, data: bytes, scope: str) -> None:
        identity = (path, hashlib.sha256(data).hexdigest())
        if identity in scanned:
            return
        scanned.add(identity)
        reasons = {"private_artifact_path"} if forbidden(path) else set()
        if len(data) > MAX_BYTES:
            reasons.add("oversize_file_requires_review")
        else:
            reasons.update(scan_bytes(path, data, known))
        for reason in sorted(reasons):
            findings.append({"scope": scope, "path": path, "reason": reason})

    paths = set(git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\0"))
    current_count = 0
    for raw in sorted(paths - {b""}):
        path = raw.decode()
        local = root / path
        if local.is_symlink():
            findings.append({"scope": "working_tree", "path": path, "reason": "symlink_requires_review"})
        elif local.is_file():
            current_count += 1
            inspect(path, local.read_bytes(), "working_tree")

    revisions = git(root, "rev-list", "--all").splitlines()
    objects: dict[tuple[str, str], str] = {}
    for revision in revisions:
        commit = revision.decode()
        for entry in git(root, "ls-tree", "-r", "-z", commit).split(b"\0"):
            if not entry:
                continue
            metadata, raw_path = entry.split(b"\t", 1)
            _, kind, oid = metadata.split()
            if kind == b"blob":
                objects.setdefault((oid.decode(), raw_path.decode()), commit)
    cache: dict[str, bytes] = {}
    for (oid, path), commit in objects.items():
        if oid not in cache:
            cache[oid] = git(root, "cat-file", "blob", oid)
        inspect(path, cache[oid], f"history:{commit[:12]}")
    # Commit/tag messages can contain sensitive values too.
    for kind, args in (
        ("commit_metadata", ("log", "--all", "--format=fuller")),
        ("tag_metadata", ("for-each-ref", "--format=%(contents)", "refs/tags")),
    ):
        for reason in sorted(scan_bytes(kind, git(root, *args), known)):
            findings.append({"scope": "history", "path": kind, "reason": reason})
    print(json.dumps({
        "working_tree_files": current_count,
        "reachable_commits": len(revisions),
        "historical_blobs": len(cache),
        "findings": findings,
        "status": "review_required" if findings else "passed_heuristic_checks",
        "limitations": "Pattern/exact-value scan only; normal Git author metadata is retained.",
    }, indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
