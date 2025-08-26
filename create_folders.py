import os
from pathlib import Path
from envpath.env import ASSETS_PATH

# All folders will be created under the assets directory
ASSETS_ROOT = ASSETS_PATH

folders = [
    ASSETS_ROOT,
    ASSETS_ROOT / "jp_fonts",
    ASSETS_ROOT / "manga" / "synthetic",
    ASSETS_ROOT / "manga" / "Manga109s" / "background",
    ASSETS_ROOT / "manga" / "Manga109s",
    ASSETS_ROOT / "manga" / "out",
]

def create_folders():
    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"Created or exists: {folder}")
        except PermissionError:
            print(f"Permission denied: {folder}")
        except Exception as e:
            print(f"Error creating {folder}: {e}")

if __name__ == "__main__":
    create_folders()
