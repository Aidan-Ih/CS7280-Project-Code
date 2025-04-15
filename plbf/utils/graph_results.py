import matplotlib.pyplot as plt

adabf_url = {'one-shot': [202, 128, 81, 46, 20], 'unif': [820459, 337305, 204160, 146676, 108895], 'zipf': [219114, 68781, 56946, 50648, 40050]}
adabf_news = {'one-shot': [211, 124, 58, 41, 33], 'unif': [7031, 1726, 864, 1452, 548], 'zipf': [77, 73, 1, 1, 77]}
# adabf_ember = {'one-shot': [], 'unif': [], 'zipf': []}

plbf_url = {'one-shot': [176, 105, 73, 30, 20], 'unif': [10970, 6563, 4456, 1821, 1257], 'zipf': [126, 6630, 45, 25837, 56]}
plbf_news = {'one-shot': [7, 5, 4, 3, 3], 'unif': [1931, 1342, 1124, 821, 821], 'zipf': [72, 72, 72, 72, 72]}
plbf_ember = {'one-shot': [4754, 3850, 3019, 2545, 1954], 'unif': [59654, 48392, 37626, 31965, 24575], 'zipf': [32523, 39466, 15066, 26068, 4364]}

aqf_url = {'one-shot': [251, 130, 75, 38, 15], 'unif': [10745, 5670, 2959, 1441, 706], 'zipf': [10820, 5687, 2937, 1468, 726]}
aqf_news = {'one-shot': [55, 22, 16, 10, 3], 'unif': [8528, 5056, 2669, 1402, 687], 'zipf': [8535, 4997, 2728, 1382, 730]}
aqf_ember = {'one-shot': [2328, 1180, 571, 288, 151], 'unif': [21706, 11176, 5597, 2815, 1391], 'zipf': [21850, 11084, 5540, 2829, 1300]}

num_rows = {'url': 162798, 'news': 35919, 'ember': 800000}
num_true_negative_url = {'one-shot': 107117, 'unif': 6580291, 'zipf': 2292307}
num_true_negative_news = {'one-shot': 17122, 'unif': 4767929, 'zipf': 7707754}
num_true_negative_ember = {'one-shot': 400000, 'unif': 5001438, 'zipf': 2518122}

url_filter_sizes = [338400, 371808, 405216, 438624, 472032]
news_filter_sizes = [86328, 94840, 103352, 111864, 120376]
ember_filter_sizes = [1340208, 1472560, 1604912, 1737264, 1869616]

# plot url data ----------------------------------------------------------------------------------------------------------

plt.plot(url_filter_sizes, [i / (float)(i + num_true_negative_url['one-shot']) for i in plbf_url['one-shot']],  label='PLBF++')
plt.plot(url_filter_sizes, [i / (float)(i + num_true_negative_url['one-shot']) for i in aqf_url['one-shot']], label='AQF')
plt.plot(url_filter_sizes, [i / (float)(i + num_true_negative_url['one-shot']) for i in adabf_url['one-shot']], label='ADA-BF')
plt.xlabel('Filter total size (bytes)')
plt.ylabel('False-positive rate ')
plt.title('FPR-Space Tradeoff on URLs (One-pass)')
plt.legend()
plt.savefig('URL_one_shot', bbox_inches='tight')
plt.clf()

plt.plot(url_filter_sizes, [i / (float)(i + num_true_negative_url['unif']) for i in plbf_url['unif']], label='PLBF++')
plt.plot(url_filter_sizes, [i / (float)(i + num_true_negative_url['unif']) for i in aqf_url['unif']], label='AQF')
plt.plot(url_filter_sizes, [i / (float)(i + num_true_negative_url['unif']) for i in adabf_url['unif']], label='ADA-BF')
plt.xlabel('Filter total size (bytes)')
plt.ylabel('False-positive rate ')
plt.title('FPR-Space Tradeoff on URLs (10M Uniform)')
plt.legend()
plt.savefig('URL_10M_unif', bbox_inches='tight')
plt.clf()

plt.plot(url_filter_sizes, [i / (float)(i + num_true_negative_url['zipf']) for i in plbf_url['zipf']], label='PLBF++')
plt.plot(url_filter_sizes, [i / (float)(i + num_true_negative_url['zipf']) for i in aqf_url['zipf']], label='AQF')
plt.plot(url_filter_sizes, [i / (float)(i + num_true_negative_url['unif']) for i in adabf_url['unif']], label='ADA-BF')
plt.xlabel('Filter total size (bytes)')
plt.ylabel('False-positive rate ')
plt.title('FPR-Space Tradeoff on URLs (10M Zipfian)')
plt.legend()
plt.savefig('URL_10M_zipf', bbox_inches='tight')
plt.clf()

# plot news data ----------------------------------------------------------------------------------------------------------

plt.plot(news_filter_sizes, [i / (float)(i + num_true_negative_news['one-shot']) for i in plbf_news['one-shot']],  label='PLBF++')
plt.plot(news_filter_sizes, [i / (float)(i + num_true_negative_news['one-shot']) for i in aqf_news['one-shot']], label='AQF')
plt.plot(news_filter_sizes, [i / (float)(i + num_true_negative_news['one-shot']) for i in adabf_news['one-shot']], label='ADA-BF')
plt.xlabel('Filter total size (bytes)')
plt.ylabel('False-positive rate ')
plt.title('FPR-Space Tradeoff on News (One-pass)')
plt.legend()
plt.savefig('News_one_shot', bbox_inches='tight')
plt.clf()

plt.plot(news_filter_sizes, [i / (float)(i + num_true_negative_news['unif']) for i in plbf_news['unif']], label='PLBF++')
plt.plot(news_filter_sizes, [i / (float)(i + num_true_negative_news['unif']) for i in aqf_news['unif']], label='AQF')
plt.plot(news_filter_sizes, [i / (float)(i + num_true_negative_news['unif']) for i in adabf_news['unif']], label='ADA-BF')
plt.xlabel('Filter total size (bytes)')
plt.ylabel('False-positive rate ')
plt.title('FPR-Space Tradeoff on News (10M Uniform)')
plt.legend()
plt.savefig('News_10M_unif', bbox_inches='tight')
plt.clf()

plt.plot(news_filter_sizes, [i / (float)(i + num_true_negative_news['zipf']) for i in plbf_news['zipf']], label='PLBF++')
plt.plot(news_filter_sizes, [i / (float)(i + num_true_negative_news['zipf']) for i in aqf_news['zipf']], label='AQF')
plt.plot(news_filter_sizes, [i / (float)(i + num_true_negative_news['unif']) for i in adabf_news['unif']], label='ADA-BF')
plt.xlabel('Filter total size (bytes)')
plt.ylabel('False-positive rate ')
plt.title('FPR-Space Tradeoff on News (10M Zipfian)')
plt.legend()
plt.savefig('News_10M_zipf', bbox_inches='tight')
plt.clf()

# plot ember data ----------------------------------------------------------------------------------------------------------

plt.plot(ember_filter_sizes, [i / (float)(i + num_true_negative_ember['one-shot']) for i in plbf_ember['one-shot']],  label='PLBF++')
plt.plot(ember_filter_sizes, [i / (float)(i + num_true_negative_ember['one-shot']) for i in aqf_ember['one-shot']], label='AQF')
# plt.plot(ember_filter_sizes, [i / (float)(i + num_true_negative_ember['one-shot']) for i in adabf_ember['one-shot']], label='ADA-BF')
plt.xlabel('Filter total size (bytes)')
plt.ylabel('False-positive rate ')
plt.title('FPR-Space Tradeoff on Ember (One-pass)')
plt.legend()
plt.savefig('ember_one_shot', bbox_inches='tight')
plt.clf()

plt.plot(ember_filter_sizes, [i / (float)(i + num_true_negative_ember['unif']) for i in plbf_ember['unif']], label='PLBF++')
plt.plot(ember_filter_sizes, [i / (float)(i + num_true_negative_ember['unif']) for i in aqf_ember['unif']], label='AQF')
# plt.plot(ember_filter_sizes, [i / (float)(i + num_true_negative_ember['unif']) for i in adabf_ember['unif']], label='ADA-BF')
plt.xlabel('Filter total size (bytes)')
plt.ylabel('False-positive rate ')
plt.title('FPR-Space Tradeoff on Ember (10M Uniform)')
plt.legend()
plt.savefig('ember_10M_unif', bbox_inches='tight')
plt.clf()

plt.plot(ember_filter_sizes, [i / (float)(i + num_true_negative_ember['zipf']) for i in plbf_ember['zipf']], label='PLBF++')
plt.plot(ember_filter_sizes, [i / (float)(i + num_true_negative_ember['zipf']) for i in aqf_ember['zipf']], label='AQF')
# plt.plot(ember_filter_sizes, [i / (float)(i + num_true_negative_ember['unif']) for i in adabf_ember['unif']], label='ADA-BF')
plt.xlabel('Filter total size (bytes)')
plt.ylabel('False-positive rate ')
plt.title('FPR-Space Tradeoff on Ember (10M Zipfian)')
plt.legend()
plt.savefig('ember_10M_zipf', bbox_inches='tight')
plt.clf()