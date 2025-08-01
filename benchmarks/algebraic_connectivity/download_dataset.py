import os
import subprocess

# Clone the repository
os.mkdir('content')
os.chdir('content')

REPO = "PowerGraph-Graph"
if os.path.isdir(REPO):
    print(f"⏭ {REPO} already exists—skipping clone")
else:
    subprocess.run([
        "git", "clone",
        "https://github.com/PowerGraph-Datasets/PowerGraph-Graph.git",
        REPO
    ], check=True)

# Download and extract the dataset
os.chdir(REPO)
zip_path = "dataset_cascades.zip"
if not os.path.isfile(zip_path):
    print("Downloading dataset...")
    subprocess.run([
        "curl", "-L", "-o", zip_path,
        "-A", "Mozilla/5.0 (X11; Linux x86_64)",
        "https://figshare.com/ndownloader/files/46619158"
    ], check=True)
else:
    print(f"{zip_path} already exists—skipping download")

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
if not os.listdir(data_dir):
    print("Unzipping dataset...")
    subprocess.run(["unzip", "-q", zip_path, "-d", data_dir], check=True)
else:
    print(f"{data_dir} already populated—skipping unzip")

print("Contents of data/:", os.listdir(data_dir))
