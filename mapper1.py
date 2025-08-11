#!/usr/bin/env python3
import sys, csv, re

reader = csv.reader(sys.stdin)
for row in reader:
    if not row:
        continue

    raw_label = (row[0] or "").strip().strip('"').lower()
    if raw_label in {"label", "sentiment"}:
        continue

    try:
        label = int(raw_label)
    except ValueError:
        continue

    # Map numeric label to text category
    sentiment = None
    if label == 1:
        sentiment = "negative"
    elif label == 2:
        sentiment = "positive"

    if sentiment and len(row) > 1:
        text = row[1].lower()
        words = re.findall(r'\w+', text)
        for word in words:
            print(f"{sentiment}:{word}\t1")
