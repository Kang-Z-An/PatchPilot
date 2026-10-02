from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tools import RepositoryTools
from .trajectory import TrajectoryWriter
from .types import Model, ToolResult


SYSTEM_PROMPT = """You are PatchPilot, a code-repair agent. Fix the reported issue in the repository.

You have at most __MAX_STEPS__ model turns. Work incrementally: search exact identifiers from the issue first, inspect their definitions and direct base classes, reproduce the issue early with run_tests when useful, form a concrete hypothesis, make the smallest justified edit, then run tests. For Python inheritance, MRO, or __slots__ issues, use inspect_python_class before manually tracing parent classes. Do not repeat equivalent searches. Reserve enough turns for an edit, run_tests, git_diff, and the final response.

The first edit-attempt deadline is turn __EDIT_DEADLINE__. Beginning with that turn, if you have not attempted an edit, only replace_text or apply_patch will be accepted. At most __MAX_INSPECTION_STREAK__ consecutive inspection calls are accepted before an edit attempt. Treat these as hard limits: act on the best-supported hypothesis instead of trying another broad search.

Prefer replace_text for a small exact edit. Use apply_patch for multi-file or structurally complex changes. Do not claim success without both a passing run_tests result and a successful git_diff after the latest edit.

For apply_patch, send only a standard unified diff with a final newline. Do not use *** Begin Patch or *** Update File directives. Example:
--- a/example.py
+++ b/example.py
@@ -1 +1 @@
-old line
+new line

Reply with exactly one JSON object and no Markdown. It must be either:
{"tool":"tool_name","arguments":{...}}
or:
{"final":"brief outcome and verification summary"}

Available tools:
__TOOLS__
"""

PROMPT_VARIANTS = (
    "default", "behavior-contract-v1", "behavior-probe-v1", "red-green-probe-v1",
    "dual-checkout-probe-v1",
)
BEHAVIOR_CONTRACT_PROMPT = """
Before editing, turn the issue into a short behavior contract in your own reasoning: identify the observable input, expected output, relevant edge cases, and the layer that owns the behavior. Distinguish the reported symptom from its likely cause; check callers and consumers before choosing an edit location. For each proposed fix, ask whether it covers all stated cases without changing unrelated behavior.

Use a focused reproduction or existing test that exercises the issue when feasible, while respecting the edit deadline. After editing, run the available tests and inspect the diff. Passing a broad regression suite alone does not prove the issue is fixed: if you could not exercise the reported behavior directly, say so in the final response instead of claiming full verification.
"""
BEHAVIOR_PROBE_PROMPT = """
When run_probe is available, use a small Python assert to check an observable behavior stated in the issue. Prefer running the same probe before and after the edit when the step budget allows. A passing probe is evidence only for its assertion; it cannot replace run_tests, git_diff, or official evaluation. Never claim broad resolution from a narrow probe.
"""
RED_GREEN_PROBE_PROMPT = """
Before editing, use run_probe with a short assertion for a concrete behavior requested in the issue. The assertion must fail on the original checkout; an import error or other execution error does not count as a red probe. The first genuine failing assertion is frozen. After editing, rerun run_probe with exactly the same code so it can turn green; changed assertions will be rejected. Both red and green are required for Agent-side verification in this variant, along with run_tests and git_diff. A passing probe is still narrow evidence, not official task resolution.
"""
DUAL_CHECKOUT_PROBE_PROMPT = """
Use run_probe_pair with a short Python assertion for an observable behavior requested in the issue. You may write the probe after editing: the tool automatically runs exactly the same code on a separate original base checkout and the current checkout. Success requires an assertion failure on the original and a pass on the current checkout; import errors and assertions that pass on both are not qualifying evidence. Inspect both outputs if it fails. After every successful edit, obtain a fresh passing pair, run_tests, and git_diff before finishing. These are all required for Agent-side verification. A passing pair proves only its assertion, not all issue edge cases or official task resolution.
"""


@dataclass(frozen=True)
class AgentResult:
    success: bool
    answer: str
    steps: int
    trajectory_path: Path
    metrics: "AgentMetrics"


@dataclass(frozen=True)
class AgentMetrics:
    tool_calls: int = 0
    tool_failures: int = 0
    invalid_actions: int = 0
    edit_attempts: int = 0
    edit_successes: int = 0
    patch_attempts: int = 0
    patch_successes: int = 0
    replace_text_attempts: int = 0
    replace_text_successes: int = 0
    inspection_calls: int = 0
    first_edit_step: int = 0
    budget_blocked_actions: int = 0
    automatic_git_diff_calls: int = 0
    protected_edit_attempts: int = 0
    empty_diff_attempts: int = 0
    test_files_modified: int = 0
    tests_run_after_latest_edit: bool = False
    probe_attempts: int = 0
    probe_successes: int = 0
    probe_attempted_after_latest_edit: bool = False
    probe_passed_after_latest_edit: bool = False
    red_probe_recorded: bool = False
    green_probe_passed: bool = False
    probe_mismatch_attempts: int = 0
    paired_probe_attempts: int = 0
    paired_probe_successes: int = 0
    paired_probe_passed_after_latest_edit: bool = False

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "tool_calls": self.tool_calls,
            "tool_failures": self.tool_failures,
            "tool_failure_rate": self._rate(self.tool_failures, self.tool_calls),
            "invalid_actions": self.invalid_actions,
            "edit_attempts": self.edit_attempts,
            "edit_successes": self.edit_successes,
            "edit_success_rate": self._rate(self.edit_successes, self.edit_attempts),
            "patch_attempts": self.patch_attempts,
            "patch_successes": self.patch_successes,
            "patch_success_rate": self._rate(self.patch_successes, self.patch_attempts),
            "replace_text_attempts": self.replace_text_attempts,
            "replace_text_successes": self.replace_text_successes,
            "replace_text_success_rate": self._rate(
                self.replace_text_successes,
                self.replace_text_attempts,
            ),
            "inspection_calls": self.inspection_calls,
            "first_edit_step": self.first_edit_step,
            "budget_blocked_actions": self.budget_blocked_actions,
            "automatic_git_diff_calls": self.automatic_git_diff_calls,
            "protected_edit_attempts": self.protected_edit_attempts,
            "empty_diff_attempts": self.empty_diff_attempts,
            "test_files_modified": self.test_files_modified,
            "tests_run_after_latest_edit": self.tests_run_after_latest_edit,
            "probe_attempts": self.probe_attempts,
            "probe_successes": self.probe_successes,
            "probe_attempted_after_latest_edit": self.probe_attempted_after_latest_edit,
            "probe_passed_after_latest_edit": self.probe_passed_after_latest_edit,
            "red_probe_recorded": self.red_probe_recorded,
            "green_probe_passed": self.green_probe_passed,
            "probe_mismatch_attempts": self.probe_mismatch_attempts,
            "paired_probe_attempts": self.paired_probe_attempts,
            "paired_probe_successes": self.paired_probe_successes,
            "paired_probe_success_rate": self._rate(self.paired_probe_successes, self.paired_probe_attempts),
            "paired_probe_passed_after_latest_edit": self.paired_probe_passed_after_latest_edit,
        }

    @staticmethod
    def _rate(successes: int, attempts: int) -> float:
        return round(successes / attempts, 4) if attempts else 0.0


class Agent:
    def __init__(
        self,
        model: Model,
        tools: RepositoryTools,
        trajectory_dir: Path,
        max_steps: int = 20,
        trajectory_metadata: dict[str, str] | None = None,
        prompt_variant: str = "default",
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(f"Unknown prompt variant: {prompt_variant}")
        self.model = model
        self.tools = tools
        self.trajectory_dir = trajectory_dir
        self.tools.ignore_diff_path(trajectory_dir)
        self.max_steps = max_steps
        self.trajectory_metadata = dict(trajectory_metadata or {})
        self.prompt_variant = prompt_variant
        self.edit_deadline = max(1, min(max_steps - 3, (max_steps * 3 + 4) // 5))
        self.max_inspection_streak = max(1, min(8, self.edit_deadline - 1))

    def run(self, issue: str) -> AgentResult:
        system_prompt = SYSTEM_PROMPT
        if self.prompt_variant != "default":
            system_prompt = system_prompt.replace(
                "Reply with exactly one JSON object and no Markdown.",
                BEHAVIOR_CONTRACT_PROMPT
                + (BEHAVIOR_PROBE_PROMPT if self.prompt_variant == "behavior-probe-v1" else "")
                + (RED_GREEN_PROBE_PROMPT if self.prompt_variant == "red-green-probe-v1" else "")
                + (DUAL_CHECKOUT_PROBE_PROMPT if self.prompt_variant == "dual-checkout-probe-v1" else "")
                + "\nReply with exactly one JSON object and no Markdown.",
            )
        messages = [
            {
                "role": "system",
                "content": system_prompt.replace(
                    "__TOOLS__", self.tools.prompt_description()
                )
                .replace("__MAX_STEPS__", str(self.max_steps))
                .replace("__EDIT_DEADLINE__", str(self.edit_deadline))
                .replace("__MAX_INSPECTION_STREAK__", str(self.max_inspection_streak)),
            },
            {
                "role": "user",
                "content": (
                    f"Issue:\n{issue}\n\n"
                    f"Step budget: {self.max_steps} total turns. "
                    f"Make the first edit attempt before turn {self.edit_deadline}; "
                    "keep enough turns to test, inspect the diff, and finish."
                ),
            },
        ]
        task_id = self.trajectory_metadata.get("task_id") or (
            "ad-hoc-" + hashlib.sha256(issue.encode("utf-8")).hexdigest()[:16]
        )
        writer = TrajectoryWriter(
            self.trajectory_dir,
            {
                "trajectory_schema_version": 2,
                "task_id": task_id,
                "source": self.trajectory_metadata.get("source", "ad_hoc"),
                "model": self.trajectory_metadata.get("model", "unknown"),
                "issue": issue,
                "repository": str(self.tools.root),
                "max_steps": self.max_steps,
                "edit_deadline": self.edit_deadline,
                "max_inspection_streak": self.max_inspection_streak,
                "prompt_variant": self.prompt_variant,
                "initial_messages": messages,
            },
        )
        tests_passed = False
        patch_applied = False
        diff_inspected = False
        tests_run_after_latest_edit = False
        counters = {
            "tool_calls": 0,
            "tool_failures": 0,
            "invalid_actions": 0,
            "edit_attempts": 0,
            "edit_successes": 0,
            "patch_attempts": 0,
            "patch_successes": 0,
            "replace_text_attempts": 0,
            "replace_text_successes": 0,
            "inspection_calls": 0,
            "first_edit_step": 0,
            "budget_blocked_actions": 0,
            "automatic_git_diff_calls": 0,
            "protected_edit_attempts": 0,
            "empty_diff_attempts": 0,
            "test_files_modified": 0,
            "tests_run_after_latest_edit": False,
            "probe_attempts": 0,
            "probe_successes": 0,
            "probe_attempted_after_latest_edit": False,
            "probe_passed_after_latest_edit": False,
            "red_probe_recorded": False,
            "green_probe_passed": False,
            "probe_mismatch_attempts": 0,
            "paired_probe_attempts": 0,
            "paired_probe_successes": 0,
            "paired_probe_passed_after_latest_edit": False,
        }
        red_probe_code: str | None = None
        red_probe_sha256: str | None = None
        paired_probe_sha256: str | None = None
        consecutive_inspections = 0
        for step in range(1, self.max_steps + 1):
            raw = self.model.complete(messages)
            writer.record("model_output", step=step, raw=raw)
            action, error = self._parse_action(raw)
            if error:
                counters["invalid_actions"] += 1
                observation = f"Invalid action: {error}. Return exactly one valid JSON object."
                writer.record(
                    "invalid_action",
                    step=step,
                    error=error,
                    observation=observation,
                )
                messages.extend([
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": observation},
                ])
                continue
            if "final" in action:
                answer = action["final"]
                if patch_applied and tests_passed and not diff_inspected:
                    automatic_diff = self.tools.execute("git_diff", {})
                    counters["tool_calls"] += 1
                    counters["automatic_git_diff_calls"] += 1
                    if not automatic_diff.ok:
                        counters["tool_failures"] += 1
                    self._record_diff_metrics(counters, automatic_diff)
                    diff_inspected = automatic_diff.ok
                    writer.record(
                        "tool_result",
                        step=step,
                        tool="git_diff",
                        arguments={},
                        ok=automatic_diff.ok,
                        output=automatic_diff.output,
                        data=automatic_diff.data,
                        observation=None,
                        automatic=True,
                    )
                counters["tests_run_after_latest_edit"] = tests_run_after_latest_edit
                verified = (
                    patch_applied
                    and tests_passed
                    and tests_run_after_latest_edit
                    and diff_inspected
                    and (
                        not counters["probe_attempted_after_latest_edit"]
                        or counters["probe_passed_after_latest_edit"]
                    )
                    and (
                        self.prompt_variant != "red-green-probe-v1"
                        or (
                            counters["red_probe_recorded"]
                            and counters["green_probe_passed"]
                        )
                    )
                    and (
                        self.prompt_variant != "dual-checkout-probe-v1"
                        or counters["paired_probe_passed_after_latest_edit"]
                    )
                )
                metrics = AgentMetrics(**counters)
                writer.record(
                    "run_finished",
                    step=step,
                    answer=answer,
                    verified=verified,
                    verification={
                        "edit_applied": patch_applied,
                        "tests_passed": tests_passed,
                        "tests_run_after_latest_edit": tests_run_after_latest_edit,
                        "diff_inspected": diff_inspected,
                        "non_empty_diff": diff_inspected,
                        "test_files_modified": counters["test_files_modified"],
                        "probe_attempted_after_latest_edit": counters["probe_attempted_after_latest_edit"],
                        "probe_passed_after_latest_edit": counters["probe_passed_after_latest_edit"],
                        "red_probe_recorded": counters["red_probe_recorded"],
                        "green_probe_passed": counters["green_probe_passed"],
                        "red_probe_sha256": red_probe_sha256,
                        "paired_probe_passed_after_latest_edit": counters["paired_probe_passed_after_latest_edit"],
                        "paired_probe_sha256": paired_probe_sha256,
                    },
                    metrics=metrics.as_dict(),
                )
                return AgentResult(verified, answer, step, writer.path, metrics)

            name = action["tool"]
            arguments = action["arguments"]
            restriction = self._budget_restriction(
                name,
                patch_applied,
                counters["edit_attempts"] > 0,
                step,
                consecutive_inspections,
            )
            if restriction is None:
                if (
                    self.prompt_variant == "red-green-probe-v1"
                    and name == "run_probe"
                    and patch_applied
                    and (
                        red_probe_code is None
                        or arguments.get("code") != red_probe_code
                    )
                ):
                    result = ToolResult(
                        False,
                        "Post-edit probe must exactly replay the first failing "
                        "pre-edit assertion; no qualifying baseline or code mismatch.",
                        {"reason": "probe_code_mismatch"},
                    )
                else:
                    result = self.tools.execute(name, arguments)
            else:
                result = ToolResult(False, restriction)
                counters["budget_blocked_actions"] += 1
            counters["tool_calls"] += 1
            if name in {
                "list_files",
                "search_code",
                "read_file",
                "inspect_python_class",
            }:
                counters["inspection_calls"] += 1
                consecutive_inspections += 1
            else:
                consecutive_inspections = 0
            expected_red = (
                self.prompt_variant == "red-green-probe-v1"
                and name == "run_probe"
                and not patch_applied
                and red_probe_code is None
                and not result.ok
                and result.data.get("outcome") == "assertion_failed"
                and isinstance(arguments.get("code"), str)
            )
            if not result.ok and not expected_red:
                counters["tool_failures"] += 1
            if name in {"apply_patch", "replace_text"}:
                edit_kind = "patch" if name == "apply_patch" else "replace_text"
                counters["edit_attempts"] += 1
                counters[f"{edit_kind}_attempts"] += 1
                if result.ok:
                    counters["edit_successes"] += 1
                    counters[f"{edit_kind}_successes"] += 1
                    if counters["first_edit_step"] == 0:
                        counters["first_edit_step"] = step
                    patch_applied = True
                    tests_passed = False
                    tests_run_after_latest_edit = False
                    counters["probe_attempted_after_latest_edit"] = False
                    counters["probe_passed_after_latest_edit"] = False
                    counters["green_probe_passed"] = False
                    counters["paired_probe_passed_after_latest_edit"] = False
                    paired_probe_sha256 = None
                    diff_inspected = False
                elif result.data.get("reason") == "protected_test_file":
                    counters["protected_edit_attempts"] += 1
            elif name == "run_tests":
                tests_passed = result.ok
                if patch_applied:
                    tests_run_after_latest_edit = True
            elif name == "run_probe":
                counters["probe_attempts"] += 1
                if result.ok:
                    counters["probe_successes"] += 1
                if result.data.get("reason") == "probe_code_mismatch":
                    counters["probe_mismatch_attempts"] += 1
                if expected_red:
                    red_probe_code = arguments["code"]
                    red_probe_sha256 = hashlib.sha256(
                        red_probe_code.encode("utf-8")
                    ).hexdigest()
                    counters["red_probe_recorded"] = True
                    writer.record(
                        "probe_red_recorded", step=step,
                        code_sha256=red_probe_sha256,
                    )
                if patch_applied:
                    counters["probe_attempted_after_latest_edit"] = True
                    counters["probe_passed_after_latest_edit"] = result.ok
                    if (
                        self.prompt_variant == "red-green-probe-v1"
                        and result.ok
                        and arguments.get("code") == red_probe_code
                    ):
                        counters["green_probe_passed"] = True
                        writer.record(
                            "probe_green_passed", step=step,
                            code_sha256=red_probe_sha256,
                        )
            elif name == "run_probe_pair":
                counters["paired_probe_attempts"] += 1
                qualified = (
                    result.ok
                    and result.data.get("red_green") is True
                    and result.data.get("baseline_unchanged") is True
                    and result.data.get("baseline_outcome") == "assertion_failed"
                    and result.data.get("candidate_outcome") == "passed"
                )
                if qualified:
                    counters["paired_probe_successes"] += 1
                if patch_applied:
                    counters["paired_probe_passed_after_latest_edit"] = qualified
                    paired_probe_sha256 = result.data.get("code_sha256") if qualified else None
                    if qualified:
                        writer.record(
                            "paired_probe_verified", step=step,
                            code_sha256=paired_probe_sha256,
                            base_commit=result.data.get("base_commit"),
                        )
            elif name == "git_diff":
                self._record_diff_metrics(counters, result)
                if result.ok and patch_applied:
                    diff_inspected = True
            remaining = self.max_steps - step
            observation = json.dumps(
                {
                    "tool": name,
                    "ok": result.ok,
                    "output": result.output,
                    "progress": {
                        "steps_used": step,
                        "steps_remaining": remaining,
                        "edit_applied": patch_applied,
                        "tests_passed_after_latest_edit": tests_passed,
                        "tests_run_after_latest_edit": tests_run_after_latest_edit,
                        "diff_inspected_after_latest_edit": diff_inspected,
                        "probe_attempted_after_latest_edit": counters["probe_attempted_after_latest_edit"],
                        "probe_passed_after_latest_edit": counters["probe_passed_after_latest_edit"],
                        "paired_probe_passed_after_latest_edit": counters["paired_probe_passed_after_latest_edit"],
                        "edit_deadline": self.edit_deadline,
                        "consecutive_inspections": consecutive_inspections,
                    },
                    "next_priority": (
                        "Use run_probe_pair to obtain an original-fail/current-pass assertion "
                        "for the reported issue after the latest edit."
                        if self.prompt_variant == "dual-checkout-probe-v1"
                        and patch_applied and tests_passed
                        and not counters["paired_probe_passed_after_latest_edit"]
                        else
                        "The post-edit issue probe failed. Recheck its assertion and "
                        "the fix; do not claim verification until a fresh probe passes."
                        if counters["probe_attempted_after_latest_edit"]
                        and not counters["probe_passed_after_latest_edit"]
                        else self._next_priority(
                            patch_applied,
                            counters["edit_attempts"] > 0,
                            tests_passed,
                            diff_inspected,
                            remaining,
                            step,
                            self.edit_deadline,
                        )
                    ),
                },
                ensure_ascii=False,
            )
            writer.record(
                "tool_result",
                step=step,
                tool=name,
                arguments=arguments,
                ok=result.ok,
                output=result.output,
                data=result.data,
                observation=observation,
            )
            messages.extend([
                {"role": "assistant", "content": raw},
                {"role": "user", "content": observation},
            ])

        answer = f"Stopped after reaching the {self.max_steps}-step limit."
        counters["tests_run_after_latest_edit"] = tests_run_after_latest_edit
        metrics = AgentMetrics(**counters)
        writer.record(
            "run_stopped",
            reason="max_steps",
            answer=answer,
            metrics=metrics.as_dict(),
        )
        return AgentResult(False, answer, self.max_steps, writer.path, metrics)

    @staticmethod
    def _record_diff_metrics(
        counters: dict[str, int | bool],
        result: ToolResult,
    ) -> None:
        if result.data.get("reason") == "empty_diff":
            counters["empty_diff_attempts"] = int(counters["empty_diff_attempts"]) + 1
        test_files = result.data.get("test_files_modified")
        if isinstance(test_files, list):
            counters["test_files_modified"] = len(test_files)

    def _budget_restriction(
        self,
        tool_name: str,
        patch_applied: bool,
        edit_attempted: bool,
        step: int,
        consecutive_inspections: int,
    ) -> str | None:
        if patch_applied:
            return None
        turns_including_current = self.max_steps - step + 1
        edit_tools = {"apply_patch", "replace_text"}
        inspection_tools = {
            "list_files",
            "search_code",
            "read_file",
            "inspect_python_class",
        }
        if step >= self.edit_deadline and not edit_attempted and tool_name not in edit_tools:
            return (
                f"Action blocked: turn {self.edit_deadline} is the first edit-attempt "
                "deadline. Use replace_text or apply_patch now with the best-supported fix."
            )
        if step >= self.edit_deadline and edit_attempted and tool_name in {
            "list_files",
            "search_code",
            "inspect_python_class",
        }:
            return (
                "Action blocked after the edit deadline: broad discovery is closed. "
                "Use one focused read_file only if needed to correct an edit, then edit again."
            )
        if (
            not edit_attempted
            and tool_name in inspection_tools
            and consecutive_inspections >= self.max_inspection_streak
        ):
            return (
                "Action blocked: the consecutive inspection limit was reached. Form a "
                "concrete hypothesis and make the smallest defensible edit now."
            )
        if turns_including_current <= 4 and tool_name in {
            "list_files",
            "search_code",
            "read_file",
        }:
            return (
                "Action blocked by the step budget: no edit has been made and only "
                f"{turns_including_current} turns remain. Make the smallest defensible "
                "replace_text or apply_patch edit now."
            )
        if turns_including_current <= 5 and tool_name in {"list_files", "search_code"}:
            return (
                "Action blocked by the step budget: broad discovery is closed. You may "
                "perform at most one focused read_file or make the best-supported edit now."
            )
        return None

    @staticmethod
    def _next_priority(
        patch_applied: bool,
        edit_attempted: bool,
        tests_passed: bool,
        diff_inspected: bool,
        steps_remaining: int,
        step: int,
        edit_deadline: int,
    ) -> str:
        if not patch_applied and not edit_attempted and step + 1 >= edit_deadline:
            return (
                "The first edit-attempt deadline is now. Make the smallest defensible "
                "replace_text or apply_patch edit next."
            )
        if not patch_applied and steps_remaining <= 4:
            return (
                "Budget critical: stop broad inspection. Make the smallest defensible "
                "edit now so turns remain for run_tests, git_diff, and final."
            )
        if patch_applied and not tests_passed:
            return "Run run_tests next unless one narrowly targeted inspection is essential."
        if tests_passed and not diff_inspected:
            return "Run git_diff next to verify the final change."
        if patch_applied and tests_passed and diff_inspected:
            return "Return the final outcome now."
        return "Continue focused diagnosis; avoid repeating searches already completed."

    @staticmethod
    def _parse_action(raw: str) -> tuple[dict[str, Any], str | None]:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            return {}, f"invalid JSON: {exc.msg}"
        if not isinstance(value, dict):
            return {}, "top-level value must be an object"
        if set(value) == {"final"} and isinstance(value["final"], str) and value["final"].strip():
            return value, None
        if set(value) != {"tool", "arguments"}:
            return {}, "expected either {final} or {tool, arguments}"
        if not isinstance(value["tool"], str) or not isinstance(value["arguments"], dict):
            return {}, "tool must be a string and arguments must be an object"
        return value, None
