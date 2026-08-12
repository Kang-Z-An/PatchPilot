from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tools import is_test_file_path


@dataclass(frozen=True)
class TestStatus:
    success: tuple[str, ...]
    failure: tuple[str, ...]

    @property
    def total(self) -> int:
        return len(self.success) + len(self.failure)

    @property
    def all_passed(self) -> bool | None:
        return not self.failure if self.total else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": len(self.success),
            "failed": len(self.failure),
            "all_passed": self.all_passed,
            "failures": list(self.failure),
        }


@dataclass(frozen=True)
class InstanceReport:
    instance_id: str
    resolved: bool
    patch_applied: bool
    target: TestStatus
    regression: TestStatus


def analyze_swebench_run(
    run_dir: Path,
    gold_report_roots: list[Path] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    summary = _load_json_object(run_dir / "summary.json", "batch summary")
    run_id = _require_string(summary, "run_id", run_dir / "summary.json")
    model = _require_string(summary, "model", run_dir / "summary.json")
    raw_cases = summary.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"Batch summary has no cases: {run_dir / 'summary.json'}")

    official_run_id = _official_run_id(run_dir, summary)
    primary_root = (
        run_dir
        / "official"
        / "logs"
        / "run_evaluation"
        / official_run_id
        / model
    )
    automatic_gold_root = run_dir / "official" / "logs" / "run_evaluation"
    gold_roots = [automatic_gold_root, *(gold_report_roots or [])]

    cases: list[dict[str, Any]] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("Every batch summary case must be a JSON object")
        instance_id = _require_string(raw_case, "instance_id", run_dir / "summary.json")
        official_status = raw_case.get("official_status", "unknown")
        if not isinstance(official_status, str):
            official_status = "unknown"

        primary_path = primary_root / instance_id / "report.json"
        primary = _load_instance_report(primary_path, instance_id, required=False)
        gold_paths = _find_gold_reports(gold_roots, instance_id, primary_path)
        gold = (
            _load_instance_report(gold_paths[-1], instance_id, required=True)
            if gold_paths
            else None
        )
        patch_path = _case_patch_path(run_dir, instance_id)
        modified_files = _patch_paths(patch_path) if patch_path is not None else []
        test_files = [path for path in modified_files if is_test_file_path(path)]

        target_passed = primary is not None and primary.target.all_passed is True
        regression_failures = (
            set(primary.regression.failure) if primary is not None else set()
        )
        candidate = (
            official_status == "unresolved"
            and target_passed
            and bool(regression_failures)
        )
        same_gold_failures = (
            candidate
            and gold is not None
            and gold.target.all_passed is True
            and set(gold.regression.failure) == regression_failures
        )
        environment_suspect = bool(same_gold_failures)
        gold_status = _gold_status(primary, gold)
        cases.append(
            {
                "instance_id": instance_id,
                "official_status": official_status,
                "detailed_report": "found" if primary is not None else "missing",
                "target_tests": (
                    primary.target.as_dict() if primary is not None else None
                ),
                "regression_tests": (
                    primary.regression.as_dict() if primary is not None else None
                ),
                "patch_applied": primary.patch_applied if primary is not None else None,
                "modified_files": modified_files,
                "test_files_modified": test_files,
                "gold_control": {
                    "status": gold_status,
                    "reports_found": len(gold_paths),
                    "target_tests": gold.target.as_dict() if gold is not None else None,
                    "regression_tests": (
                        gold.regression.as_dict() if gold is not None else None
                    ),
                },
                "environment_suspect_candidate": candidate,
                "environment_suspect": environment_suspect,
                "environment_reason": _environment_reason(
                    candidate,
                    environment_suspect,
                    gold,
                ),
            }
        )

    official = summary.get("official")
    if not isinstance(official, dict):
        official = {}
    reports_found = [case for case in cases if case["detailed_report"] == "found"]
    target_cases_passed = [
        case
        for case in reports_found
        if case["target_tests"] is not None
        and case["target_tests"]["all_passed"] is True
    ]
    target_cases_failed = [
        case
        for case in reports_found
        if case["target_tests"] is not None
        and case["target_tests"]["all_passed"] is False
    ]
    environment_suspects = [case for case in cases if case["environment_suspect"]]
    candidates_without_confirmation = [
        case
        for case in cases
        if case["environment_suspect_candidate"] and not case["environment_suspect"]
    ]
    test_edit_cases = [case for case in cases if case["test_files_modified"]]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "model": model,
        "official": {
            "status": official.get("status"),
            "resolved": official.get("resolved"),
            "unresolved": official.get("unresolved"),
            "errors": official.get("errors"),
            "resolve_rate": official.get("resolve_rate"),
        },
        "audit": {
            "total_instances": len(cases),
            "detailed_reports_found": len(reports_found),
            "instances_with_all_target_tests_passed": len(target_cases_passed),
            "instances_with_target_test_failures": len(target_cases_failed),
            "target_test_results_missing": len(cases) - len(reports_found),
            "total_regression_test_failures": sum(
                int(case["regression_tests"]["failed"])
                for case in reports_found
                if case["regression_tests"] is not None
            ),
            "instances_with_test_file_changes": len(test_edit_cases),
            "environment_suspect_instances": [
                case["instance_id"] for case in environment_suspects
            ],
            "unconfirmed_environment_candidates": [
                case["instance_id"] for case in candidates_without_confirmation
            ],
            "score_policy": (
                "Official resolved counts and resolve_rate are copied unchanged. "
                "Environment findings are audit annotations, not adjusted scores."
            ),
        },
        "cases": cases,
    }


def write_analysis(
    analysis: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(analysis), encoding="utf-8")
    return json_path, markdown_path


def _load_instance_report(
    path: Path,
    instance_id: str,
    *,
    required: bool,
) -> InstanceReport | None:
    if not path.is_file():
        if required:
            raise ValueError(f"SWE-bench instance report does not exist: {path}")
        return None
    data = _load_json_object(path, "SWE-bench instance report")
    raw = data.get(instance_id)
    if not isinstance(raw, dict):
        raise ValueError(f"Instance {instance_id!r} is missing from {path}")
    tests = raw.get("tests_status")
    if not isinstance(tests, dict):
        raise ValueError(f"tests_status is missing for {instance_id!r} in {path}")
    return InstanceReport(
        instance_id=instance_id,
        resolved=bool(raw.get("resolved", False)),
        patch_applied=bool(raw.get("patch_successfully_applied", False)),
        target=_parse_test_status(tests.get("FAIL_TO_PASS"), path, "FAIL_TO_PASS"),
        regression=_parse_test_status(
            tests.get("PASS_TO_PASS"), path, "PASS_TO_PASS"
        ),
    )


def _parse_test_status(value: Any, path: Path, name: str) -> TestStatus:
    if not isinstance(value, dict):
        raise ValueError(f"{name} is missing from {path}")
    success = value.get("success")
    failure = value.get("failure")
    if not isinstance(success, list) or not all(isinstance(item, str) for item in success):
        raise ValueError(f"{name}.success must be a list of strings in {path}")
    if not isinstance(failure, list) or not all(isinstance(item, str) for item in failure):
        raise ValueError(f"{name}.failure must be a list of strings in {path}")
    return TestStatus(tuple(success), tuple(failure))


def _official_run_id(run_dir: Path, summary: dict[str, Any]) -> str:
    result_path = run_dir / "official" / "result.json"
    if result_path.is_file():
        result = _load_json_object(result_path, "official evaluation result")
        run_id = result.get("run_id")
        if isinstance(run_id, str) and run_id.strip():
            return run_id.strip()
    return f"patchpilot-{_require_string(summary, 'run_id', run_dir / 'summary.json')}"


def _find_gold_reports(
    roots: list[Path],
    instance_id: str,
    primary_path: Path,
) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if root.is_file() and root.name == "report.json":
            candidates = [root]
        elif root.is_dir():
            candidates = list(root.glob(f"**/gold/{instance_id}/report.json"))
            if root.name == instance_id and root.parent.name == "gold":
                candidates.append(root / "report.json")
        else:
            continue
        for candidate in candidates:
            if candidate.is_file() and candidate.resolve() != primary_path.resolve():
                found.add(candidate.resolve())
    return sorted(found)


def _case_patch_path(run_dir: Path, instance_id: str) -> Path | None:
    case_path = run_dir / "cases" / f"{instance_id}.json"
    if case_path.is_file():
        case = _load_json_object(case_path, "batch case result")
        raw = case.get("patch_path")
        if isinstance(raw, str) and raw.strip():
            path = Path(raw)
            if not path.is_absolute():
                path = run_dir / path
            if path.is_file():
                return path.resolve()
    candidates = sorted((run_dir / "artifacts" / instance_id).glob("*/model.patch"))
    return candidates[-1].resolve() if candidates else None


def _patch_paths(path: Path) -> list[str]:
    try:
        patch = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read model patch {path}: {exc}") from exc
    paths: set[str] = set()
    for header in re.findall(r"^(?:---|\+\+\+)\s+([^\t\n]+)", patch, re.MULTILINE):
        if header == "/dev/null":
            continue
        normalized = header[2:] if header.startswith(("a/", "b/")) else header
        paths.add(normalized)
    return sorted(paths)


def _gold_status(
    primary: InstanceReport | None,
    gold: InstanceReport | None,
) -> str:
    if gold is None:
        return "not_available"
    if gold.resolved:
        return "resolved"
    if primary is not None and set(gold.regression.failure) == set(
        primary.regression.failure
    ):
        return "same_regression_failures"
    return "different_failures"


def _environment_reason(
    candidate: bool,
    environment_suspect: bool,
    gold: InstanceReport | None,
) -> str | None:
    if environment_suspect:
        return (
            "Target tests passed, and the gold control failed the identical non-empty "
            "set of regression tests."
        )
    if candidate and gold is None:
        return "Target tests passed but regression tests failed; no gold control is available."
    if candidate:
        return "Target tests passed but the gold control did not reproduce identical failures."
    return None


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read {description} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {description} {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description.capitalize()} must be a JSON object: {path}")
    return value


def _require_string(data: dict[str, Any], key: str, path: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing non-empty {key!r} in {path}")
    return value.strip()


def _render_markdown(analysis: dict[str, Any]) -> str:
    official = analysis["official"]
    audit = analysis["audit"]
    lines = [
        f"# SWE-bench run audit: {analysis['run_id']}",
        "",
        f"- Model: `{analysis['model']}`",
        (
            f"- Official result: {official.get('resolved')}/{audit['total_instances']} "
            f"resolved ({official.get('resolve_rate')})"
        ),
        (
            "- Target-test cases passed: "
            f"{audit['instances_with_all_target_tests_passed']}/{audit['total_instances']}"
        ),
        (
            "- Environment-suspect instances: "
            f"{len(audit['environment_suspect_instances'])}"
        ),
        (
            "- Instances with test-file changes: "
            f"{audit['instances_with_test_file_changes']}"
        ),
        "",
        "> The official score is preserved. Environment findings are audit annotations only.",
        "",
        "| Instance | Official | Target | Regression failures | Test files changed | Gold | Environment suspect |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for case in analysis["cases"]:
        target = case["target_tests"]
        target_text = (
            f"{target['passed']}/{target['total']}" if target is not None else "missing"
        )
        regression = case["regression_tests"]
        regression_failures = regression["failed"] if regression is not None else "missing"
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{case['instance_id']}`",
                    case["official_status"],
                    target_text,
                    str(regression_failures),
                    str(len(case["test_files_modified"])),
                    case["gold_control"]["status"],
                    "yes" if case["environment_suspect"] else "no",
                )
            )
            + " |"
        )
    lines.extend(("", "## Environment findings", ""))
    suspects = [case for case in analysis["cases"] if case["environment_suspect"]]
    if not suspects:
        lines.append("No instance met the strict environment-suspect rule.")
    else:
        for case in suspects:
            lines.append(f"- `{case['instance_id']}`: {case['environment_reason']}")
    return "\n".join(lines) + "\n"
