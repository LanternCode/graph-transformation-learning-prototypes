import glob
import os
import urllib.request

# List of .col files to download
FILES = [
    "DSJC125.1.col"
]

# DIMACS GitHub mirror
BASE_URL = "https://raw.githubusercontent.com/BartMassey/instances/master/"
DEST_DIR = "graph_coloring_benchmark/graphs"

# Step 1: Clean existing files
os.makedirs(DEST_DIR, exist_ok=True)
for f in glob.glob(os.path.join(DEST_DIR, "*.col")):
    print(f"Deleting old file: {f}")
    os.remove(f)

# Step 2: Download each file
for fname in FILES:
    url = BASE_URL + fname
    dst = os.path.join(DEST_DIR, fname)
    try:
        print(f"Downloading {fname}...")
        urllib.request.urlretrieve(url, dst)
        size = os.path.getsize(dst)
        if size == 0:
            print(f"{fname} is empty — removing.")
            os.remove(dst)
        else:
            print(f"{fname} downloaded — {size} bytes.")
    except Exception as e:
        print(f"Failed to download {fname}: {e}")
