import numpy as np
import pandas as pd
import argparse
from Bloom_filter import BloomFilter


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                    help="path of the dataset")
parser.add_argument('--threshold_min', action="store", dest="min_thres", type=float, required=True,
                    help="Minimum threshold for positive samples")
parser.add_argument('--threshold_max', action="store", dest="max_thres", type=float, required=True,
                    help="Maximum threshold for positive samples")
parser.add_argument('--size_of_Ada_BF', action="store", dest="R_sum", type=int, required=True,
                    help="size of the Ada-BF")




results = parser.parse_args()
DATA_PATH = results.data_path
min_thres = results.min_thres
max_thres = results.max_thres
R_sum = results.R_sum


# DATA_PATH = './URL_data.csv'
# min_thres = 0.5
# max_thres = 0.9
# R_sum = 200000



'''
Load the data and select training data
'''
data = pd.read_csv(DATA_PATH)
data.loc[data['label'] == 'malicious', 'label'] = 1
data.loc[data['label'] == 'benign', 'label'] = 0
negative_sample = data.loc[(data['label']==0)]
positive_sample = data.loc[(data['label']==1)]
train_negative = negative_sample.sample(frac = 0.3)


def Find_Optimal_Parameters(max_thres, min_thres, R_sum, train_negative, positive_sample):
    FP_opt = train_negative.shape[0]

    for threshold in np.arange(min_thres, max_thres+10**(-6), 0.01):
        url = positive_sample.loc[(positive_sample['score'] <= threshold),'key']
        n = len(url)
        bloom_filter = BloomFilter(n, R_sum)
        bloom_filter.insert(url)
        ML_positive = train_negative.loc[(train_negative['score'] > threshold),'key']
        bloom_negative = train_negative.loc[(train_negative['score'] <= threshold),'key']
        BF_positive = bloom_filter.test(bloom_negative, single_key=False)
        FP_items = sum(BF_positive) + len(ML_positive)
        if FP_opt > FP_items:
            FP_opt = FP_items
            thres_opt = threshold
            bloom_filter_opt = bloom_filter

    return bloom_filter_opt, thres_opt


'''
Implement learned Bloom filter
'''
if __name__ == '__main__':
    size_of_model = 180000
    FPR = []
    for i in range(20):
        total_size = size_of_model + ((i + 1) * 10000)
        filter_size = total_size - size_of_model
        print(filter_size)
        bloom_filter_opt, thres_opt = Find_Optimal_Parameters(max_thres, min_thres, filter_size, train_negative, positive_sample)

        ### Test URLs
        ML_positive = negative_sample.loc[(negative_sample['score'] > thres_opt), 'key']
        bloom_negative = negative_sample.loc[(negative_sample['score'] <= thres_opt), 'key']
        score_negative = negative_sample.loc[(negative_sample['score'] < thres_opt), 'score']
        BF_positive = bloom_filter_opt.test(bloom_negative, single_key = False)
        FP_items = sum(BF_positive) + len(ML_positive)
        
        print('FPR: ', (FP_items / len(negative_sample)))
        FPR.append(float(np.round((FP_items / len(negative_sample)), 5)))
        
    print(FPR)