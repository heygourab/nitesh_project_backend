# app/model/train_sms_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import os
from pathlib import Path # Add this import

# Import custom preprocessing function
from preprocessing import preprocess_text

# Define base directories relative to the script
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent

# Load dataset
df = pd.read_csv(
    ROOT_DIR / "dataset" / "SMSSpamCollection.csv", # Use Path object for joining
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Preprocess raw text
df["cleaned"] = df["message"].apply(preprocess_text)

# Encode labels: 'ham'->0, 'spam'->1
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned"], df["label_num"], test_size=0.2, random_state=42, stratify=df["label_num"]
)

# Build pipeline: TF-IDF + Naive Bayes
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=3000)),  
    ("clf", MultinomialNB()),                         
])
pipeline.fit(X_train, y_train)

# Evaluate accuracy (optional)
accuracy = pipeline.score(X_test, y_test)
print(f"SMS Model Accuracy: {accuracy:.2%}")

# Serialize pipeline
model_path = SCRIPT_DIR / "spam_classifier_sms.pkl" # Save in the same directory as the script
joblib.dump(pipeline, model_path)
print(f"Saved SMS spam model to {model_path}")
#
