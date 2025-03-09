import pandas as pd

files = ["ndocs.csv", "mddocs.csv"]
files = ["data/" + f for f in files]
all_df = [pd.read_csv(f) for f in files]
combined_df = pd.concat(all_df, ignore_index=True)

combined_df.to_csv("data/aggregate.csv", index=False)