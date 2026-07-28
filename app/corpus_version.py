from datetime import datetime
from pathlib import Path

_VERSION_FILE = Path(__file__).resolve().parent.parent / "data" / "corpus_version.txt"


def bump() -> str:
    version = datetime.now().isoformat(timespec="microseconds")  # 2026-07-26T14:32:07.123456
    _VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    _VERSION_FILE.write_text(version)
    return version


def current() -> str:
    return _VERSION_FILE.read_text().strip() if _VERSION_FILE.exists() else "none"


if __name__ == "__main__":
    print("version:", bump())