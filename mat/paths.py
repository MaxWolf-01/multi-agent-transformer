from pathlib import Path


class Paths:
    ROOT = Path(__file__).parent.parent
    PROJECT = ROOT / "src"
    VIDEOS = ROOT / "videos"
    CKPTS = ROOT / "checkpoints"
    DATA = ROOT / "data"
    LOGS = ROOT / "logs"
