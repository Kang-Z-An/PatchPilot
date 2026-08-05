from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from patchpilot.tools import RepositoryTools


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


if __name__ == "__main__":
    unittest.main()
