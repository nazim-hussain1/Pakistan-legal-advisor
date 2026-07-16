import pandas as pd
import re

df = pd.read_csv("fyp_cleaned_dataset.csv", low_memory=False).fillna("")
col = df["clean_text"].astype(str)

BODY_START, SCHEDULE_START = 328676, 334922
missing = ['47', '91', '96', '103', '104', '105', '106', '133', '134', '135',
           '167', '171', '197', '214', '216', '223', '229', '247', '256',
           '260', '262', '270', '272', '152A', '212A', '212B', '224A', '203CC']

for art in missing:
    pattern = re.compile(rf"(?:^|\s){re.escape(art)}\.\s")
    hits = []
    for r in range(BODY_START, SCHEDULE_START):
        if pattern.search(col.iloc[r]):
            hits.append(r)
    print(f"Article {art}: {len(hits)} raw hits in body -> {hits[:3]}")
    if hits:
        print(f"   sample: {col.iloc[hits[0]][:120]}")