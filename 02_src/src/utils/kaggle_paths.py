"""Path helpers that work both locally and on Kaggle."""
import os

def is_kaggle() -> bool:
    return os.path.exists('/kaggle')

def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def input_dir(name: str = '') -> str:
    """Read-only data. Kaggle: /kaggle/input/<name>. Local: 01_data/<name>."""
    base = '/kaggle/input' if is_kaggle() else os.path.join(_repo_root(), '01_data')
    return os.path.join(base, name)

def working_dir(name: str = '') -> str:
    """Writable outputs. Kaggle: /kaggle/working. Local: 05_experiments/<name>."""
    base = '/kaggle/working' if is_kaggle() else os.path.join(_repo_root(), '05_experiments')
    os.makedirs(os.path.join(base, name), exist_ok=True)
    return os.path.join(base, name)
