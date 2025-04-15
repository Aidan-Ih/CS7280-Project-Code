import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# unhashed_querysets = ["unhashed_news_10M", "unhashed_news_100M", "unhashed_url_10M", "unhashed_url_100M"]
# hashed_querysets = ["hashed_news_10M", "hashed_news_100M", "hashed_url_10M", "hashed_url_100M"]
# unhashed_querysets = ["unhashed_unif_10M_news", "unhashed_unif_100M_news", "unhashed_unif_10M_url", "unhashed_unif_100M_url",
#                       "unhashed_zipf_10M_news", "unhashed_zipf_100M_news", "unhashed_zipf_10M_url", "unhashed_zipf_100M_url"]
hashed_querysets = ["hashed_unif_10M_news", "hashed_unif_100M_news", "hashed_unif_10M_url", "hashed_unif_100M_url",
                    "hashed_zipf_10M_news", "hashed_zipf_100M_news", "hashed_zipf_10M_url", "hashed_zipf_100M_url", "hashed_zipf_10M_ember", "hashed_zipf_100M_ember"]

# for unhashed_queryset in unhashed_querysets:
#     df = pd.read_csv('data/query_indices/' + unhashed_queryset + '.csv')
#     indexes = df.iloc[:, 1]
#     plt.hist(indexes, bins=50, log=True)
#     plt.xlabel('Index')
#     plt.ylabel('Frequency')
#     plt.savefig('queries_' + unhashed_queryset + '.png')
#     plt.clf()

for hashed_queryset in hashed_querysets:
    df = pd.read_csv('data/query_indices/' + hashed_queryset + '.csv')
    indexes = df.iloc[:, 1]
    plt.hist(indexes, bins=50, log=True)
    plt.xlabel('Index')
    plt.ylabel('Frequency')
    plt.savefig('queries_' + hashed_queryset + '.png')
    plt.clf()