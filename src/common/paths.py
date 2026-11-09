"""Project assets are resolved relative to the repository, independently of cwd."""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = PROJECT_ROOT / "configs"
SIMULATION_DIR = CONFIGS / "simulation"
DATA = PROJECT_ROOT / "data"
RUNS = PROJECT_ROOT / "runs"
WEIGHTS = PROJECT_ROOT / "weights"
THIRD_PARTY = PROJECT_ROOT / "third_party"

def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
