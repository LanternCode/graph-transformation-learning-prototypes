import glob
import os
import urllib.request
from collections.abc import Sequence

FILES = [
    "DSJC125.1.col",
]

BASE_URL = "https://raw.githubusercontent.com/BartMassey/instances/master/"
DEST_DIR = "dimacs_graphs"


def clean_existing_col_files(dest_dir: str = DEST_DIR) -> None:
    """
    Delete existing DIMACS .col files from a destination directory.

    Args:
        dest_dir: Directory containing DIMACS .col files to delete.

    Returns:
        None. Matching files are removed from disk.
    """
    os.makedirs(dest_dir, exist_ok=True)
    for path in glob.glob(os.path.join(dest_dir, "*.col")):
        print(f"Deleting old file: {path}")
        os.remove(path)


def download_col_file(
    filename: str,
    base_url: str = BASE_URL,
    dest_dir: str = DEST_DIR,
    overwrite: bool = False,
) -> str:
    """
    Download one DIMACS .col file into the dataset directory.

    Args:
        filename: Name of the .col file to download from the mirror.
        base_url: Base URL of the DIMACS file mirror.
        dest_dir: Local directory where the file should be stored.
        overwrite: Whether to replace an existing local file.

    Returns:
        Path to the downloaded or already-existing local file.
    """
    os.makedirs(dest_dir, exist_ok=True)
    url = base_url + filename
    dst = os.path.join(dest_dir, filename)

    if os.path.isfile(dst) and not overwrite:
        print(f"{filename} already exists — skipping download")
        return dst

    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, dst)
    size = os.path.getsize(dst)
    if size == 0:
        os.remove(dst)
        raise ValueError(f"{filename} downloaded as an empty file")

    print(f"{filename} downloaded — {size} bytes.")
    return dst


def download_dataset(
    files: Sequence[str] = FILES,
    dest_dir: str = DEST_DIR,
    clean: bool = False,
    overwrite: bool = False,
) -> list[str]:
    """
    Download the configured DIMACS dataset files.

    Args:
        files: Sequence of .col filenames to download.
        dest_dir: Local directory where graph files should be stored.
        clean: Whether to delete existing .col files before downloading.
        overwrite: Whether to replace files that already exist.

    Returns:
        List of local file paths for downloaded or reused files.
    """
    if clean:
        clean_existing_col_files(dest_dir)

    downloaded_paths = []
    for filename in files:
        try:
            downloaded_paths.append(
                download_col_file(
                    filename=filename,
                    base_url=BASE_URL,
                    dest_dir=dest_dir,
                    overwrite=overwrite,
                )
            )
        except Exception as exc:
            print(f"Failed to download {filename}: {exc}")

    return downloaded_paths


def main() -> list[str]:
    """
    Run the default DIMACS dataset download workflow.

    Args:
        None.

    Returns:
        List of local file paths for downloaded or reused graph files.
    """
    return download_dataset()


if __name__ == "__main__":
    main()
