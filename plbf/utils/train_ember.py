from ember_import.ember.ember import create_metadata, read_vectorized_features
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import csv

# create the training and testing metadata
create_metadata("data/ember")

print("Obtaining data")
# use the Ember imported code base to convert downloaded Ember json into training and testing sets
x_train, y_train, x_test, y_test = read_vectorized_features("data/ember", feature_version=1)

# Remove all the unlabelled rows (only found in the training data)
train_rows = (y_train != -1)
malicious_rows = (y_test == 1)

malicious_y_test = y_test[malicious_rows]
malicious_x_test = x_test[malicious_rows]

# combine the training and testing set into one overall dataset.
# as described in PLBF, the training set consists of the original training set
# (without unlabelled files) combined with the malicious files from the test set
total_x_train = np.vstack((x_train[train_rows], malicious_x_test))
total_y_train = np.concatenate((y_train[train_rows], malicious_y_test))

print("Training")
# Initialize a classifier
n_estimators = 60
max_leaves = 20
modelname = 'ember_model'
clf = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaves, random_state=42)
classifier = clf.fit(total_x_train, total_y_train) # Have the model learn from the training data

# check the resulting accuracy of the classifier
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# the size of the classifier can be found in the pickle file
with open('models/' + modelname + '.pkl', 'wb') as f:
    pickle.dump(classifier, f)
size_bytes = os.path.getsize('models/' + modelname + '.pkl')
print(size_bytes)

train_x_classify = pd.read_csv("data/ember_vectorized_x_train_rm_unlabel.csv", index_col=0)
scores = clf.predict_proba(train_x_classify)[:, 1]
train_metadata_no_unlabel = pd.read_csv("data/train_metadata_rm_unlabel.csv")
train_metadata_no_unlabel["score"] = scores

test_scores = clf.predict_proba(x_test)[:, 1]
test_metadata = pd.read_csv("data/ember/test_metadata.csv")
test_metadata["score"] = test_scores

# create a unified df
combined_metadata = pd.concat([train_metadata_no_unlabel, test_metadata], ignore_index=True)
combined_metadata.to_csv("combined_ember_metadata.csv")