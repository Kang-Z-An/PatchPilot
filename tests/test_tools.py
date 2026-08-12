from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from patchpilot.tools import RepositoryTools, is_test_file_path


class RepositoryToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        (self.root / "maths.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (self.root / "test_maths.py").write_text(
            "import unittest\nfrom maths import add\n\n"
            "class MathsTest(unittest.TestCase):\n"
            "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n",
            encoding="utf-8",
        )
        self.tools = RepositoryTools(self.root, "python -m unittest -q")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_search_read_patch_and_test(self) -> None:
        search = self.tools.execute("search_code", {"query": "return a - b"})
        self.assertTrue(search.ok)
        self.assertIn("maths.py:2", search.output)

        read = self.tools.execute("read_file", {"path": "maths.py"})
        self.assertTrue(read.ok)
        self.assertIn("return a - b", read.output)

        patch = """--- a/maths.py
+++ b/maths.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""
        applied = self.tools.execute("apply_patch", {"patch": patch})
        self.assertTrue(applied.ok, applied.output)
        tests = self.tools.execute("run_tests", {})
        self.assertTrue(tests.ok, tests.output)

    def test_paths_cannot_escape_repository(self) -> None:
        result = self.tools.execute("read_file", {"path": "../secret.txt"})
        self.assertFalse(result.ok)
        self.assertIn("escapes repository", result.output)

        malicious = """--- a/../secret.txt
+++ b/../secret.txt
@@ -0,0 +1 @@
+nope
"""
        result = self.tools.execute("apply_patch", {"patch": malicious})
        self.assertFalse(result.ok)
        self.assertIn("Unsafe patch path", result.output)

    def test_inspect_python_class_traces_slots_through_ancestors(self) -> None:
        (self.root / "hierarchy.py").write_text(
            "class Printable:\n"
            "    pass\n\n"
            "class Basic(Printable):\n"
            "    __slots__ = ()\n\n"
            "class Symbol(Basic):\n"
            "    __slots__ = ('name',)\n",
            encoding="utf-8",
        )

        result = self.tools.execute(
            "inspect_python_class",
            {"class_name": "Symbol", "max_depth": 4},
        )

        self.assertTrue(result.ok, result.output)
        payload = json.loads(result.output)
        symbol = payload["definitions"][0]
        basic = symbol["bases"][0]["definitions"][0]
        printable = basic["bases"][0]["definitions"][0]
        self.assertTrue(symbol["declares_slots"])
        self.assertTrue(basic["declares_slots"])
        self.assertFalse(printable["declares_slots"])
        self.assertEqual(printable["path"], "hierarchy.py")

    def test_replace_text_requires_one_exact_match(self) -> None:
        result = self.tools.execute(
            "replace_text",
            {
                "path": "maths.py",
                "old_text": "return a - b",
                "new_text": "return a + b",
            },
        )

        self.assertTrue(result.ok, result.output)
        self.assertIn("return a + b", (self.root / "maths.py").read_text(encoding="utf-8"))
        missing = self.tools.execute(
            "replace_text",
            {
                "path": "maths.py",
                "old_text": "return a - b",
                "new_text": "return a + b",
            },
        )
        self.assertFalse(missing.ok)
        self.assertIn("found 0", missing.output)

    def test_identifies_conventional_test_paths(self) -> None:
        protected = (
            "tests/test_math.py",
            "package/test/helpers.py",
            "testing/test_capture.py",
            "package/tests/unit/check.py",
            "test_root.py",
            "package/widget_test.py",
            "package/conftest.py",
        )
        source = (
            "package/testing_utils.py",
            "package/contest/result.py",
            "package/testimony.py",
            "package/widget_tests.py",
            "src/check.py",
        )

        for path in protected:
            with self.subTest(path=path):
                self.assertTrue(is_test_file_path(path))
        for path in source:
            with self.subTest(path=path):
                self.assertFalse(is_test_file_path(path))

    def test_replace_text_rejects_protected_test_file(self) -> None:
        tools = RepositoryTools(
            self.root,
            "python -m unittest -q",
            protect_test_files=True,
        )
        before = (self.root / "test_maths.py").read_text(encoding="utf-8")

        result = tools.execute(
            "replace_text",
            {
                "path": "test_maths.py",
                "old_text": "self.assertEqual(add(2, 3), 5)",
                "new_text": "self.assertEqual(add(2, 3), -1)",
            },
        )

        self.assertFalse(result.ok)
        self.assertIn("Protected test file", result.output)
        self.assertEqual(
            (self.root / "test_maths.py").read_text(encoding="utf-8"), before
        )
        self.assertIn("Fix production code only", tools.prompt_description())

    def test_apply_patch_rejects_multi_file_test_edit_atomically(self) -> None:
        tools = RepositoryTools(
            self.root,
            "python -m unittest -q",
            protect_test_files=True,
        )
        patch = """--- a/maths.py
+++ b/maths.py
@@ -2 +2 @@
-    return a - b
+    return a + b
--- a/test_maths.py
+++ b/test_maths.py
@@ -6 +6 @@
-        self.assertEqual(add(2, 3), 5)
+        self.assertEqual(add(2, 3), -1)
"""

        result = tools.execute("apply_patch", {"patch": patch})

        self.assertFalse(result.ok)
        self.assertIn("test_maths.py", result.output)
        self.assertIn(
            "return a - b", (self.root / "maths.py").read_text(encoding="utf-8")
        )

    def test_test_edit_protection_can_be_disabled_explicitly(self) -> None:
        tools = RepositoryTools(
            self.root,
            "python -m unittest -q",
            protect_test_files=False,
        )

        result = tools.execute(
            "replace_text",
            {
                "path": "test_maths.py",
                "old_text": "self.assertEqual(add(2, 3), 5)",
                "new_text": "self.assertEqual(add(2, 3), -1)",
            },
        )

        self.assertTrue(result.ok, result.output)

    def test_git_diff_rejects_empty_change(self) -> None:
        self._commit_baseline()

        result = self.tools.execute("git_diff", {})

        self.assertFalse(result.ok)
        self.assertEqual(result.data["reason"], "empty_diff")
        self.assertFalse(result.data["has_diff"])
        self.assertEqual(result.data["changed_files"], [])

    def test_git_diff_reports_test_files_and_enforces_final_protection(self) -> None:
        self._commit_baseline()
        (self.root / "test_maths.py").write_text(
            (self.root / "test_maths.py")
            .read_text(encoding="utf-8")
            .replace("self.assertEqual(add(2, 3), 5)", "self.assertEqual(add(2, 3), -1)"),
            encoding="utf-8",
        )
        unprotected = self.tools.execute("git_diff", {})
        protected = RepositoryTools(
            self.root,
            "python -m unittest -q",
            protect_test_files=True,
        ).execute("git_diff", {})

        self.assertTrue(unprotected.ok, unprotected.output)
        self.assertEqual(unprotected.data["test_files_modified"], ["test_maths.py"])
        self.assertFalse(protected.ok)
        self.assertEqual(protected.data["reason"], "protected_final_diff")
        self.assertEqual(protected.data["test_files_modified"], ["test_maths.py"])

    def test_apply_patch_normalizes_wrapper_and_final_newline(self) -> None:
        patch = """*** Begin Patch
--- a/maths.py
+++ b/maths.py
@@ -2 +2 @@
-    return a - b
+    return a + b
*** End Patch"""

        result = self.tools.execute("apply_patch", {"patch": patch})

        self.assertTrue(result.ok, result.output)
        self.assertIn("return a + b", (self.root / "maths.py").read_text(encoding="utf-8"))

    def test_codex_update_directive_returns_actionable_error(self) -> None:
        patch = """*** Begin Patch
*** Update File: maths.py
@@
-    return a - b
+    return a + b
*** End Patch"""

        result = self.tools.execute("apply_patch", {"patch": patch})

        self.assertFalse(result.ok)
        self.assertIn("use replace_text", result.output)

    def test_git_diff_includes_new_files_but_excludes_patchpilot_artifacts(self) -> None:
        (self.root / "new_module.py").write_text("VALUE = 1\n", encoding="utf-8")
        artifact_dir = self.root / ".patchpilot"
        artifact_dir.mkdir()
        (artifact_dir / "trajectory.jsonl").write_text("private trace\n", encoding="utf-8")

        result = self.tools.execute("git_diff", {})

        self.assertTrue(result.ok, result.output)
        self.assertIn("new_module.py", result.output)
        self.assertNotIn("trajectory.jsonl", result.output)
        self.assertNotIn("private trace", result.output)

    def _commit_baseline(self) -> None:
        subprocess.run(["git", "add", "--all"], cwd=self.root, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=PatchPilot Test",
                "-c",
                "user.email=test@patchpilot.local",
                "commit",
                "-qm",
                "baseline",
            ],
            cwd=self.root,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
