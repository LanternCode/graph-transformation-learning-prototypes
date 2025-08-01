import os

# Download the dataset
if not os.path.exists("higgs-retweet_network.edgelist"):
    if not os.path.exists("higgs-retweet_network.edgelist.gz"):
        print("Downloading dataset...")
        os.system("wget https://snap.stanford.edu/data/higgs-retweet_network.edgelist.gz")
    else:
        print("Compressed file already exists.")

    print("Decompressing...")
    os.system("gunzip higgs-retweet_network.edgelist.gz")
else:
    print("Decompressed dataset already exists.")
