from __future__ import annotations

import unittest

from scripts.check_release import forbidden, scan_bytes


class ReleaseChecksTest(unittest.TestCase):
    def test_detects_keys_and_exact_local_values_without_emitting_values(self) -> None:
        token = b"sk-" + b"a" * 32
        self.assertEqual(scan_bytes("example.py", token, []), {"api_key"})
        private_value = b"private-provider-value"
        findings = scan_bytes("example.py", private_value, [private_value])
        self.assertEqual(findings, {"known_local_secret"})
        self.assertNotIn(private_value.decode(), str(findings))

    def test_url_fixture_exception_is_limited_to_reserved_hosts(self) -> None:
        prefix = b"https://" + b"user:pass@"
        self.assertEqual(scan_bytes("fixture.json", prefix + b"example.com", []), set())
        self.assertEqual(scan_bytes("fixture.json", prefix + b"real-provider.com", []), {"credential_url"})
        self.assertEqual(scan_bytes("fixture.json", prefix + b"example.com.evil.org", []), {"credential_url"})

    def test_blocks_private_artifacts_but_keeps_env_template(self) -> None:
        for path in (".env", ".env.production", ".patchpilot/trace.jsonl", "nested/id.key", "checkpoints/model.pt"):
            self.assertTrue(forbidden(path))
        self.assertFalse(forbidden(".env.example"))
        self.assertFalse(forbidden("benchmarks/cases/example/case.json"))
