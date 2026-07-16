import pandas as pd

df = pd.read_csv("fyp_cleaned_dataset.csv", low_memory=False).fillna("")
text = "\n".join(df.astype(str).agg(" ".join, axis=1).tolist())

for art in ["10A", "19A", "25A"]:
    count = text.count(f"Article {art}")
    print(f"Article {art}: appears {count} times in raw text")