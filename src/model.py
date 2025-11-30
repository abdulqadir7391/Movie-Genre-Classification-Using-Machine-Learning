import pandas as pd
import ast

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ---------------------------
# Load dataset
# ---------------------------
data = pd.read_csv("data/movies_metadata.csv", dtype={"id": str})
data = data[['overview', 'genres']]
data = data.dropna(subset=['overview', 'genres'])

# ---------------------------
# Parse genres properly
# ---------------------------
def parse_genres(x):
    try:
        genres_list = ast.literal_eval(x)
        return [g['name'] for g in genres_list if 'name' in g]
    except:
        return []

data['genres'] = data['genres'].apply(parse_genres)
data = data[data['genres'].map(len) > 0]

# Features and labels
X = data['overview']
y = data['genres']

# ---------------------------
# Binarize labels for multi-label classification
# ---------------------------
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y)

# ---------------------------
# Train/test split
# ---------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ---------------------------
# TF-IDF
# ---------------------------
tfidf = TfidfVectorizer(stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ---------------------------
# Multi-label Logistic Regression
# ---------------------------
model = OneVsRestClassifier(LogisticRegression(max_iter=1200))
model.fit(X_train_tfidf, Y_train)

# ---------------------------
# Evaluation
# ---------------------------
Y_pred = model.predict(X_test_tfidf)

# Subset accuracy (exact match)
subset_acc = accuracy_score(Y_test, Y_pred)
print("Subset Accuracy:", subset_acc)

# Macro-averaged F1-score
macro_f1 = f1_score(Y_test, Y_pred, average='macro')
print("Macro F1-Score:", macro_f1)

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred, target_names=mlb.classes_))

# ---------------------------
# Prediction Function
# ---------------------------
def predict_genres(text):
    vector = tfidf.transform([text])
    predicted = model.predict(vector)[0]
    return [genre for genre, val in zip(mlb.classes_, predicted) if val == 1]

# Example prediction:
print("\nExample prediction:")
new_plot = "A boy travels through space to fight evil robots and discover his destiny."
print("Text:", new_plot)
print("Predicted genres:", predict_genres(new_plot))
