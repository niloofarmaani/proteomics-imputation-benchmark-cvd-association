from pathlib import Path
import os

# /.../project_root/code
CODE_DIR = Path(__file__).resolve().parent

# /.../project_root
PROJECT_ROOT = CODE_DIR.parent

# Optional: override where data lives (e.g. on scratch)
BIOBANK_ROOT = Path(os.environ.get("IMPUTATION_DATA_ROOT", str(PROJECT_ROOT / "work"))).resolve()

# User-configurable experiments directory name
EXPERIMENTS_NAME = "Experiments"


# Directories that matter
EXPERIMENTS_DIR   = BIOBANK_ROOT / EXPERIMENTS_NAME
PREPROCESSING_DIR = CODE_DIR / "preprocessing"
IMPUTATION_DIR    = CODE_DIR / "imputation"


def experiments_dir(columns_num):
    """
    Folder: <experiments_name>/<columns_num>
    """
    return EXPERIMENTS_DIR / str(columns_num)


def fraction_dir(columns_num, fraction, missingness):
    """
    Folder: <experiments_name>/<columns_num>/fraction_X_missingness_Y
    """
    return experiments_dir(columns_num) / f"fraction_{fraction}_missingness_{missingness}"


def set_biobank_root(path):
    """
    Optional runtime override of BIOBANK_ROOT.
    """
    global BIOBANK_ROOT, EXPERIMENTS_DIR
    BIOBANK_ROOT = Path(path).resolve()
    EXPERIMENTS_DIR = BIOBANK_ROOT / EXPERIMENTS_NAME


def set_experiments_name(name):
    """
    Change only the experiments directory name.
    """
    global EXPERIMENTS_NAME, EXPERIMENTS_DIR
    EXPERIMENTS_NAME = name
    EXPERIMENTS_DIR = BIOBANK_ROOT / EXPERIMENTS_NAME
