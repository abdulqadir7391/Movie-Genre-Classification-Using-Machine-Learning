# 🎬 Movie Genre Classification Using Machine Learning

This project predicts the genre of a movie by analyzing its **plot summary** using **Natural Language Processing (NLP)** and **Logistic Regression**.

The model uses TF-IDF to convert text into numerical vectors and trains a machine learning classifier to identify genre patterns.

---

## 🚀 Project Overview

Movie metadata contains useful fields like *plot summary*, *genres*, *popularity*, *runtime*, etc.  
For this Week-1 ML project, we use only:

| Column      | Use |
|------------|-----|
| `overview` | Model input (plot summary text) |
| `genres`   | Model output (labels) |

Genres are extracted from JSON-like nested structures, cleaned, and formatted for classification.

---

## 🧠 Machine Learning Pipeline

The workflow:

1. **Load dataset**
2. **Clean and preprocess text**
3. **Extract features using TF-IDF**
4. **Train a Logistic Regression model**
5. **Predict genres from test data**
6. **Evaluate accuracy**
7. **Predict genres for new unseen movie plots**

---

## 📂 Dataset

- File: `movies_metadata.csv`  
- Total rows: **1779**
- Example columns used:
  - `overview` – Plot summary of the movie (text)
  - `genres` – List of genre dictionaries (string representation)


## 📂 File Structure

movie-genre-classifier/
│
├── data/
│ └── movies_metadata.csv # dataset file
│
├── src/
│ └── model.py # main ML script
│
├── notebooks/
│ └── MovieGenrePrediction.ipynb # optional Jupyter notebook
│
├── README.md
├── requirements.txt

└── .gitignore
