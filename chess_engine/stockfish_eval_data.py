import pandas as pd
import fsspec
from huggingface_hub import hf_hub_url

local_file = "stockfish_evaluations_1mill.pkl"
url = "hf://datasets/bingbangboom/stockfish-evaluations/stockfish_evaluations.jsonl"

# full data set
"""
print("Loading data from remote")
df = pd.read_json(url, lines=True)
df.to_pickle(local_file)
print("Loaded data from remote")
"""

# 5 million rows
"""
print("Loading data from remote")
df = pd.read_json(url, lines=True, nrows=5000000)
df.to_pickle(local_file)
print("Loaded data from remote")
"""

"""
# 12 million rows
print("Loading data from remote")
df = pd.read_json(url, lines=True, nrows=12000000)
df.to_pickle(local_file)
print("Loaded data from remote")
"""

# 1 million rows
print("Loading data from remote")
df = pd.read_json(url, lines=True, nrows=1000000)
df.to_pickle(local_file)
print("Loaded data from remote")