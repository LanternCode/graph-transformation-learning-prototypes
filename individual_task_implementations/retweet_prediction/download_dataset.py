import subprocess
from pathlib import Path

DATA_URL = "https://snap.stanford.edu/data/higgs-retweet_network.edgelist.gz"
COMPRESSED_PATH = Path("higgs-retweet_network.edgelist.gz")
DECOMPRESSED_PATH = Path("higgs-retweet_network.edgelist")


def download_dataset(url: str = DATA_URL,
                     compressed_path: Path = COMPRESSED_PATH,
                     decompressed_path: Path = DECOMPRESSED_PATH) -> Path:
    """
    Download and decompress the Higgs retweet dataset if it is not already present.

    Args:
        url: URL of the compressed SNAP edge-list file.
        compressed_path: Local path where the compressed ``.gz`` file is stored.
        decompressed_path: Local path where the decompressed edge-list file is stored.

    Returns:
        Path to the decompressed edge-list file.
    """
    if decompressed_path.exists():
        print(f"Decompressed dataset already exists at {decompressed_path}.")
        return decompressed_path

    if not compressed_path.exists():
        print("Downloading dataset...")
        subprocess.run(["wget", "-O", str(compressed_path), url], check=True)
    else:
        print(f"Compressed file already exists at {compressed_path}.")

    print("Decompressing dataset...")
    subprocess.run(["gunzip", "-f", str(compressed_path)], check=True)
    return decompressed_path


def main() -> None:
    """
    Run the dataset download workflow as a command-line script.

    Args:
        None.

    Returns:
        None.
    """
    download_dataset()


if __name__ == "__main__":
    main()
