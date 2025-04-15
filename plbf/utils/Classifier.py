import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import csv

# Trains a classifier, then returns an updated dataset file name with appended scores, as well as the model's size in bytes
def train_classifier(filename, n_estimators, max_leaves, modelname='model'):
    # Find the data to train on
    df = pd.read_csv('data/' + filename + '.csv')
    X = []
    if (filename == 'News_data'):
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['text'])
    else:
        X = df.drop(columns=['label'])
    Y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=26)

    clf = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaves, random_state=42)

    classifier = clf.fit(X_train, y_train) # Have the model learn from the training data

    # Create a new output file with scores
    output_filename = "data/" + filename + '_learned.csv'
    if os.path.exists(output_filename):
        os.remove(output_filename)
    df["score"] = classifier.predict_proba(X)[:,1] # Probability of being assigned 1
    df.to_csv(output_filename, index = True)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    with open('models/' + modelname + '.pkl', 'wb') as f:
        pickle.dump(classifier, f)

    size_bytes = os.path.getsize('models/' + modelname + '.pkl')

    return output_filename, size_bytes



filename, size_bytes = train_classifier('News_data', 10, 5)
print(size_bytes)