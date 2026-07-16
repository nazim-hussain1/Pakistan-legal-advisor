import pandas as pd
import re

df = pd.read_csv("fyp_cleaned_dataset.csv", low_memory=False).fillna("")
print("Columns:", list(df.columns))
print("Shape:", df.shape)

combined = df.astype(str).agg(" ".join, axis=1)
text = "\n".join(combined.tolist())

# How much of the corpus mentions the suspected foreign document?
oic_mentions = len(re.findall(r"Islamic Conference|Organization of the Islamic", text, re.IGNORECASE))
const_mentions = len(re.findall(r"Constitution of (the Islamic Republic of )?Pakistan", text, re.IGNORECASE))
print(f"'Islamic Conference' mentions: {oic_mentions}")
print(f"'Constitution of Pakistan' mentions: {const_mentions}")

# Check if there's a column that identifies source document
for col in df.columns:
    sample = df[col].astype(str).str.lower()
    if sample.str.contains("islamic conference").any():
        print(f"Column '{col}' contains OIC references — check this column's unique values:")
        print(df.loc[sample.str.contains("islamic conference"), col].unique()[:5])