import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PLAYBACK_SCRIPT = _PROJECT_ROOT / "hardware" / "playback.py"


def start_mock_stream():
    return subprocess.Popen(
        [sys.executable, str(_PLAYBACK_SCRIPT), "--source", "synthetic"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def stop_mock_stream(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def is_running(proc):
    return proc is not None and proc.poll() is None
