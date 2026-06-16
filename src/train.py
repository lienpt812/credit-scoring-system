from __future__ import annotations

import json

from src.modeling import train_and_evaluate


def main() -> None:
    report = train_and_evaluate()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
