# How to Train Your Filter: A Comparison of Learned and Adaptive Filters

This repository contains a compilation of implementations for **[Ada-BF](https://github.com/DAIZHENWEI/Ada-BF)**, **[FastPLBF](https://github.com/atsukisato/FastPLBF)**, and **[AdaptiveQF](https://github.com/splatlab/adaptiveqf)** filters. We also provide code for processing the [Fake News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), [Malicious URLs](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset), and [Ember](https://github.com/elastic/ember) datasets for use with this filter.

Ultimately, this repository provides the tools to perform a comparison of the different filters under different datasets and different query distributions.

## Starting with the code
The **AdaptiveQF** is implemented in C, while **Ada-BF** and **FastPLBF** are implemented in Python. Some additional work was done to make sure all implementations could work on the same datsets and queries.

To prepare the **AdaptiveQF**, you first need to make sure you have [SplinterDB](https://splinterdb.org/) set up. Then, we can run `make` to create the necessary executables.

## Processing Datasets and Training Models

For the datasets, to ensure they are in a standardized format for all filters, we chose to create updated CSV files for each dataset which contain the key, the score, and the true label. All the learned filters require this information, while the adaptive filter simply requires the key and the true label.

For **URL** and **News** data:
- To obtain datasets:
    - Take **URL** and **News** data from Kaggle
- To train models:
    - URL: (insert Aidan info here)
    - News: (insert Aidan info here)

For **Ember** data:
- Run `python3 plbf/utils/train_ember.py` to train a model and obtain a file with keys/scores/labels

## Obtaining Querysets
Creating the queries involves multiple steps and scripts from different folders. The end result will be a CSV File where the second column is an index describing the row of the key in the original dataset, and an (optional) CSV File containing index-count pairs.
- First, run `make` to access the `write_queries` executable, which you can use to create a binary containing integers representing query indices. `write_queries` takes the number of queries and an optional Zipfian constant as arguments (without the constant, it will assume it must generate a uniform distribution of queries instead).
    - For example, to create a uniform distribution binary file with 10M queries, run `./adaptiveqf/write_queries 10000000`
    - For example, to create a zipfian distribution binary file with 10M queries, run `./adaptiveqf/write_queries 10000000 1.5`
- Next, we convert these query binaries into CSVs for convenience. The scripts for these are inside the FASTPLBF folder
    - To convert the binary into a CSV where the second column is a query index, run `python3 plbf/utils/convert_query_binaries.py`
    - Then, if the files are too large to process, run `python3 plbf/utils/shorten_indexes.py` to obtain a CSV File containing index-count pairs.


## Running Queries
At this point, we should have trained models, scored datasets, and filters available. We can now run the tests using each queryset.

### **AdaptiveQF** tests:
The **AdaptiveQF** takes into account `q`, where there are 2<sup>`q`</sup> slots in the filter, and `r`, describing the size of the fingerprint in bits. Generally, we set `q` such that the number of slots in the filter is above the number of elements in the dataset.
- `test_one_pass` will perform the one-pass test (where every element in the dataset is queried once) for the given dataset, taking the number of elements and the path to the dataset as arguments. For example, to run the one-pass test on the Ember dataset, you can use `./adaptiveqf/test_one_pass 20 7 800000 datasets/combined_ember_metadata.csv`
- `test_distribution` will perform queries based on the given query indexes, taking the number of queries, path to dataset, and path to queries as arguments. For example, to run 10M Zipfian queries on the Ember dataset, you can use `./adaptiveqf/test_distribution 20 7 10000000 datasets/combined_ember_metadata.csv datasets/query_indices/hashed_zipf_10M_ember.csv`

### **PLBF** tests:
The **PLBF** takes into account `N`, the number of segments to split the score distribution into, `k`, the number of regions (and thus the number of underlying filters), and `M`, the memory in bytes we allow for the underlying filters. We set `M` to match the size of the **AdaptiveQF** to allow for memory-efficiency comparisons.

`FastPLBF_M_dist.py` will perform queries based on the given query index file, leaving out a query index file indicates that you prefer to do the one-pass test instead.
- For example, to run the one-pass test on the Malicious URLs dataset, you can use `python3 plbf/src/PLBFs/FastPLBF_M_dist.py --data_path data/malicious_url_scores.csv --N 1000 --k 5 --M 299069`
- For example, to run 10M Uniform queries on the Malicious URLs dataset, you can use `python3 plbf/src/PLBFs/FastPLBF_M_dist.py --data_path data/malicious_url_scores.csv --query_path data/query_indices/hashed_unif_10M_url.csv --N 1000 --k 5 --M 299069`

### **Ada-BF** tests:
The **Ada-BF** takes into account 'k_min, k_max' which define the minimum and maximum number of segments to split the score distribution, and 'c_min, c_max' which define a test range for c which we will optimize. 

'ada_bf_indexes_query.py' will perform queries based on the given query index file. Leaving out the query index ifle indicates that you prefer to do the one-pass test instead. 
- For example, to run the one-pass test on the Malicious URL's dataset, you can use `python ada_bf_indexes_query.py --data_path --data_path data/malicious_url_scores.csv --size_of_Ada_BF 20000  --num_group_min 8  --num_group_max 12  --c_min 1.6  --c_m`
- For example, to run 10M Uniform queries on the Malicious URLs dataset, you can use  `python ada_bf_indexes_query.py --data_path --data_path data/malicious_url_scores.csv --query_path data/query_indices/hashed_unif_10M_url.csv --size_of_Ada_BF 20000  --num_group_min 8  --num_group_max 12  --c_min 1.6  --c_m`