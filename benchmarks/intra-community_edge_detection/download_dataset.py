import urllib.request
import gzip
import shutil
import os

# Download and extract the SNAP Facebook combined dataset
url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
compressed_file = "facebook_combined.txt.gz"
extracted_file = "facebook_combined.txt"

# Download
if not os.path.exists(compressed_file):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, compressed_file)

# Extract
if not os.path.exists(extracted_file):
    print("Extracting dataset...")
    with gzip.open(compressed_file, 'rb') as f_in:
        with open(extracted_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
