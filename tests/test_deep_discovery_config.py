"""Rollout-flag validation for adaptive deep discovery."""

import importlib
import os
import unittest
from unittest.mock import patch

import config


class DeepDiscoveryConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("DEEP_DISCOVERY_MODE", None)
        importlib.reload(config)

    def test_defaults_off(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEEP_DISCOVERY_MODE", None)
            self.assertEqual(importlib.reload(config).DEEP_DISCOVERY_MODE, "off")

    def test_accepts_apply_case_insensitively(self) -> None:
        with patch.dict(os.environ, {"DEEP_DISCOVERY_MODE": " APPLY "}):
            self.assertEqual(importlib.reload(config).DEEP_DISCOVERY_MODE, "apply")

    def test_rejects_unknown_mode(self) -> None:
        with patch.dict(os.environ, {"DEEP_DISCOVERY_MODE": "shadow"}):
            with self.assertRaisesRegex(ValueError, "DEEP_DISCOVERY_MODE"):
                importlib.reload(config)


if __name__ == "__main__":
    unittest.main()
