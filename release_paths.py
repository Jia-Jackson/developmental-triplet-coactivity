from pathlib import Path

ROOT = Path(__file__).resolve().parent

def repository_path(relative):
    return ROOT / relative
