"""Pytest configuration for path setup."""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# Make tests reproducible by ignoring machine-specific profile/tool env config.
# Without this, importing config during test collection can pick up a developer's
# local .env and fail before tests run.
os.environ["REACHY_MINI_SKIP_DOTENV"] = "1"
os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
os.environ.pop("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY", None)
os.environ.pop("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY", None)


# Mock sounddevice for environments without PortAudio library
# This allows tests to run without actual audio hardware
class MockOutputStream:
    """Mock sounddevice OutputStream for testing."""

    def __init__(self, *args, **kwargs):
        self.latency = 0.1
        self._started = False

    def start(self):
        self._started = True

    def stop(self):
        self._started = False

    def close(self):
        self._started = False

    def write(self, data):
        # Simulate successful write
        pass

    def abort(self):
        # Simulate abort for interrupt handling
        pass


class MockDefaultDevice:
    """Mock sounddevice.default.device tuple."""

    def __getitem__(self, index):
        # Return default input/output device indices
        return 0


class MockSoundDevice:
    """Mock sounddevice module for testing without audio hardware."""

    OutputStream = MockOutputStream
    default_device = MockDefaultDevice()

    @staticmethod
    def query_devices(kind=None):
        if kind == "output":
            return {"name": "Mock Device", "max_output_channels": 2}
        return [{"name": "Mock Device", "max_output_channels": 2, "max_input_channels": 2}]

    # Make 'default' an object with 'device' attribute
    default = type("Default", (), {"device": default_device})()


# Install mock before any imports
sys.modules["sounddevice"] = MockSoundDevice()
