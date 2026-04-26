import datetime
import subprocess
from pathlib import Path

from config import OBSIDIAN_URL


def _run(cmd):
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"action failed: {cmd[0]} ({e})")


def open_obsidian_daily():
    _run(["open", "-a", "Obsidian"])
    _run(["open", OBSIDIAN_URL])


def take_screenshot():
    out = Path.home() / "Desktop" / f"shot_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
    _run(["screencapture", str(out)])

ACTIONS = {
    "open_obsidian_daily": open_obsidian_daily,
    "take_screenshot": take_screenshot,
}
