import numpy as np
import pandas as pd
from sklearn.utils import murmurhash3_32
from random import randint
import argparse



def hashfunc(m):
    ss = randint(1, 99999999)
    def hash_m(x):
        return murmurhash3_32(x,seed=ss)%m
    return hash_m


'''
Class for Standard Bloom filter
'''
class BloomFilter():
    def __init__(self, n, hash_len):
        self.n = n
        self.hash_len = int(hash_len)
        if (self.n > 0) & (self.hash_len > 0):
            self.k = max(1,int(self.hash_len/n*0.6931472))
        elif (self.n==0):
            self.k = 1
        self.h = []
        for i in range(self.k):
            self.h.append(hashfunc(self.hash_len))
        self.table = np.zeros(self.hash_len, dtype=int)
    def insert(self, key):
        if self.hash_len == 0:
            raise SyntaxError('cannot insert to an empty hash table')
        for i in key:
            for j in range(self.k):
                t = self.h[j](i)
                self.table[t] = 1
    # def test(self, key):
    #     test_result = 0
    #     match = 0
    #     if self.hash_len > 0:
    #         for j in range(self.k):
    #             t = self.h[j](key)
    #             match += 1*(self.table[t] == 1)
    #         if match == self.k:
    #             test_result = 1
    #     return test_result

    def test(self, keys, single_key = True):
        if single_key:
            test_result = 0
            match = 0
            if self.hash_len > 0:
                for j in range(self.k):
                    t = self.h[j](keys)
                    match += 1 * (self.table[t] == 1)
                if match == self.k:
                    test_result = 1
        else:
            test_result = np.zeros(len(keys))
            ss=0
            if self.hash_len > 0:
                for key in keys:
                    match = 0
                    for j in range(self.k):
                        t = self.h[j](key)
                        match += 1*(self.table[t] == 1)
                    if match == self.k:
                        test_result[ss] = 1
                    ss += 1
        return test_result

'''Run Bloom filter'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                        help="path of the dataset")
    parser.add_argument('--size_of_Ada_BF', action="store", dest="R_sum", type=int, required=True,
                        help="size of the Ada-BF")

    results = parser.parse_args()
    DATA_PATH = results.data_path
    R_sum = results.R_sum
    
    data = pd.read_csv(DATA_PATH)
    data.loc[data['label'] == 'malicious', 'label'] = 1
    data.loc[data['label'] == 'benign', 'label'] = 0

    negative_sample = data.loc[(data['label'] == 0)]
    positive_sample = data.loc[(data['label'] == 1)]

    url = positive_sample['key']
    n = len(url)
    size_of_model = 180000
    FPR = []
    for i in range(30):
        size = size_of_model + (i * 10000)
        bloom_filter = BloomFilter(n, size)
        bloom_filter.insert(url)
        url_negative = negative_sample['key']
        n1 = bloom_filter.test(url_negative, single_key=False)
        print('False positive items: ', sum(n1))
        print('FPR: ', sum(n1) / len(negative_sample))
        FPR.append(float(np.round((sum(n1) / len(negative_sample)), 5)))
        
    print ([size_of_model + (i * 10000) for i in range(20)])
    print(FPR)
