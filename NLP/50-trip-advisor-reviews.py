#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

ORIGINAL DATASET: https://arxiv.org/abs/2205.01759


"""

import pandas as pd
df=pd.read_csv("/fast-drive/a-tripadvisor-dataset-for-nlp-tasks/a-tripadvisor-dataset-for-nlp-tasks/London_reviews.csv")
print(df.columns)
df.loc[:50,['review_full']].rename(columns={'review_full':'Text'}).to_csv("50-trip-advisor-reviews.csv", index=False)
