import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

#setup data
fake_df = pd.read_csv('./archive/Fake.csv')
real_df = pd.read_csv('./archive/True.csv')
fake_df['label'] = 1
real_df['label'] = 0
combined_df = pd.concat([fake_df, real_df], ignore_index=True)


#print(f"Fake news samples: {len(fake_df)}") - 23481
#print(f"Real news samples: {len(real_df)}") - 21417
#print(f"Total samples: {len(combined_df)}") - 44898
#print("\nColumns in dataset:", combined_df.columns.tolist())
#print("\nSubject distribution:")
#print(combined_df['subject'].value_counts())


combined_df = combined_df.sample(frac=1, random_state=9012).reset_index(drop=True)
X = combined_df['title'] + " " + combined_df['text']
y = combined_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Create a pipeline with TF-IDF vectorization and Random Forest classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=1000, 
        ngram_range=(1, 1),
        stop_words='english',
        min_df=2,  
        max_df=0.95 
    )),
    ('clf', RandomForestClassifier(
        n_estimators=15,
        max_leaf_nodes=10
    ))
])

print('fitting model')
pipeline.fit(X_train, y_train)

#evaluate model
print('cross evaluating')
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")

y_pred = pipeline.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
#plt.show()

import joblib
model_filename = 'fake_news_classifier.pkl'
joblib.dump(pipeline, model_filename)
print(f"\nModel saved as {model_filename}")

def generate_prediction_csv(model, data_df, output_file='predictions.csv'):
    result_df = data_df.copy()
    
    X_test_data = result_df['title'] + " " + result_df['text']
    result_df['predicted_class'] = model.predict(X_test_data)

    probabilities = model.predict_proba(X_test_data)
    result_df['prediction_score'] = [prob[1] for prob in probabilities]
    
    output_df = result_df[['title', 'label', 'prediction_score']]
    
    output_df.to_csv(output_file, index=False)
    print(f"\nPrediction CSV saved as '{output_file}' with {len(output_df)} articles")
    
    return


generate_prediction_csv(
    model=pipeline, 
    data_df=combined_df.iloc[X_test.index],  #get test set
    output_file='article_predictions.csv')
