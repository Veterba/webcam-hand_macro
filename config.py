import os
from dotenv import load_dotenv

load_dotenv()

GESTURES = ["idle", "open_obsidian", "screenshot"]

SEQ_LEN = 30
NUM_FEATURES = 63

CONFIDENCE_THRESHOLD = 0.85
COOLDOWN_SECONDS = 2.0

ACTION_MAP = {
    "open_obsidian": "open_obsidian_daily",
    "screenshot": "take_screenshot",
}

OBSIDIAN_VAULT = os.getenv("OBS_VAULT")
OBSIDIAN_URL = f"obsidian://advanced-uri?vault={OBSIDIAN_VAULT}&commandid=daily-notes%3Agoto-today"
