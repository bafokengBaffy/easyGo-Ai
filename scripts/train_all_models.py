from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def run(cmd: list[str]):
    print(f"> {' '.join(map(str, cmd))}")
    subprocess.check_call(cmd, cwd=ROOT)

def main():
    # generate datasets
    py = sys.executable
    run([py, str(ROOT / "scripts" / "generate_datasets.py")])
    # train models
    run([py, "-m", "src.models.rider_churn_prediction.trainer"])
    run([py, "-m", "src.models.rider_ltv.trainer"])
    run([py, "-m", "src.models.driver_eta.trainer"])
    run([py, "-m", "src.models.driver_acceptance.trainer"])

if __name__ == '__main__':
    main()
