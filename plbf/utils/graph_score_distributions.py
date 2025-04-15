import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('data/combined_ember_metadata.csv')
true_rows = df.loc[df['label'] == 0]
false_rows = df.loc[df['label'] == 1]
true_scores = true_rows['score']
false_scores = false_rows['score']
plt.hist([true_scores, false_scores], bins=30, stacked=False, color=['blue', 'red'])
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend(['Non-keys', 'Keys'])
plt.savefig('figures/ember_distribution.png', bbox_inches='tight')
plt.clf()

df = pd.read_csv('data/malicious_url_scores.csv')
true_rows = df.loc[df['type'] == "benign"]
false_rows = df.loc[df['type'] == "malicious"]
true_scores = true_rows['prediction_score']
false_scores = false_rows['prediction_score']
plt.hist([true_scores, false_scores], bins=30, stacked=False, color=['blue', 'red'])
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend(['Non-keys', 'Keys'])
plt.savefig('figures/url_distribution.png', bbox_inches='tight')
plt.clf()

df = pd.read_csv('data/fake_news_predictions.csv')
true_rows = df.loc[df['label'] != 1]
false_rows = df.loc[df['label'] == 1]
true_scores = true_rows['prediction_score']
false_scores = false_rows['prediction_score']
plt.hist([true_scores, false_scores], bins=30, stacked=False, color=['blue', 'red'])
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend(['Non-keys', 'Keys'])
plt.savefig('figures/news_distribution.png', bbox_inches='tight')
plt.clf()