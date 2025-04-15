# This file is used to convert a binary file with query indexes into a csv with the index on each row.

import numpy as np
import pandas as pd
import random

a = 584141
b = 108881
p = 479001599

def hash_to_range(elements, a, b, p, max):
    return (a * elements + b) % p % max

dataset_num_rows = 44906

querysets = ["zipf_10M_ember", "zipf_100M_ember", "unif_10M_ember", "unif_100M_ember",
             "zipf_10M_url", "zipf_100M_url", "unif_10M_url", "unif_100M_url",
             "zipf_10M_news", "zipf_100M_news", "unif_10M_news", "unif_100M_news"]

for queryset in querysets:
    binary_name = "data/" + queryset + ".bin"
    f = open(binary_name, "r") 
    elements = np.fromfile(f, dtype=np.uint32)
    if "unif" in queryset:
        df.to_csv("hashed_" + queryset + ".csv")
    else:
        pd.DataFrame(elements).to_csv("data/query_indices/unhashed_" + queryset + ".csv")
        max_range = elements.shape[0]
        if "news" in queryset:
            max_range = 35919
        elif "url" in queryset:
            max_range = 162798
        elif "ember" in queryset:
            max_range = 800000
        hashed_values = hash_to_range(elements, a, b, p, max_range)
        df = pd.DataFrame(hashed_values)
        df.sort_values(by=df.columns[0], inplace=True)
        df.to_csv("data/query_indices/hashed_" + queryset + ".csv")
    f.close()