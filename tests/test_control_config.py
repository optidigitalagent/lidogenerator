import importlib
import os
import unittest
from unittest.mock import patch

import config


class ControlConfigTests(unittest.TestCase):
    def setUp(self):
        self.environ = patch.dict(os.environ, {}, clear=False)
        self.environ.start()
        for name in (
            "LEAD_GENERATOR_CONTROL_ENABLED",
            "LEAD_GENERATOR_CONTROL_TOKEN",
            "PORT",
        ):
            os.environ.pop(name, None)

    def tearDown(self):
        self.environ.stop()
        importlib.reload(config)

    def test_disabled_by_default(self):
        module = importlib.reload(config)
        self.assertFalse(module.LEAD_GENERATOR_CONTROL_ENABLED)
        self.assertEqual(8080, module.PORT)

    def test_enabled_requires_strong_token(self):
        os.environ["LEAD_GENERATOR_CONTROL_ENABLED"] = "true"
        with self.assertRaisesRegex(ValueError, "at least 32"):
            importlib.reload(config)
        os.environ["LEAD_GENERATOR_CONTROL_TOKEN"] = "x" * 32
        self.assertTrue(importlib.reload(config).LEAD_GENERATOR_CONTROL_ENABLED)

    def test_port_is_strictly_validated(self):
        os.environ["PORT"] = "not-a-port"
        with self.assertRaisesRegex(ValueError, "PORT must be an integer"):
            importlib.reload(config)
        os.environ["PORT"] = "70000"
        with self.assertRaisesRegex(ValueError, "PORT must be between"):
            importlib.reload(config)
