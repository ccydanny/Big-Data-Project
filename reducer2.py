#!/usr/bin/env python3
import sys
from collections import defaultdict, Counter

word_counts = defaultdict(Counter)

# Read mapper output
for line in sys.stdin:
    parts = line.strip().split("\t")
    if len(parts) != 2:
        continue

    key, val = parts
    try:
        count = int(val)
    except ValueError:
        continue

    if ":" not in key:
        continue
    sentiment, word = key.split(":", 1)
    word_counts[sentiment][word] += count

# Output top 30 words for each sentiment
for sentiment, counter in word_counts.items():
    print(f"Top 30 words for {sentiment}:")
    for word, cnt in counter.most_common(30):
        print(f"{word}\t{cnt}")