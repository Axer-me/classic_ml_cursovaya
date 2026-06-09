# Запуск всех этапов курсовой работы.

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    "eda.py",
    "models.regression_ic50",
    "models.regression_cc50",
    "models.regression_si",
    "models.classification_ic50_median",
    "models.classification_cc50_median",
    "models.classification_si_median",
    "models.classification_si_gt8",
    "generate_report.py",
]


def run_step(name: str) -> None:
    print("\n" + "=" * 60)
    print(f"Запуск: {name}")

    if name.endswith(".py"):
        cmd = [sys.executable, str(ROOT / name)]
    else:
        cmd = [sys.executable, "-m", name]

    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        print(f"Ошибка при выполнении {name}")
        sys.exit(result.returncode)


def main() -> None:
    for script in SCRIPTS:
        run_step(script)
    print("\nВсе этапы выполнены успешно.")


if __name__ == "__main__":
    main()
