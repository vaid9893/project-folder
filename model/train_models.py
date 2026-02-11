import pandas as pd
import numpy as np
import os
import joblib

# Importing the 6 required models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Preprocessing and Evaluation Metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,matthews_corrcoef

def train_and_save_all_models():
    # 1. Create 'model' directory if it doesn't exist
    if not os.path.exists('model'):
        os.makedirs('model')
        print("Created 'model/' directory.")

    # 2. Load Dataset
    # Ensure your 'creditcard_2023.csv' is in the root project folder
    if not os.path.exists('creditcard_2023.csv'):
        print("Error: 'creditcard_2023.csv' not found. Please place it in the root folder.")
        return

    print("Loading dataset...")
    df = pd.read_csv('creditcard_2023.csv')

    # Basic Cleaning (The 2023 dataset is usually clean, but we drop ID for safety)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # 3. Features and Target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # 4. Preprocessing: Scaling
    # V1-V28 are PCA-scaled, but 'Amount' needs scaling for LR and KNN
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    
    # Save scaler for the Streamlit app to use on new inputs
    joblib.dump(scaler, 'model/scaler.pkl')

    # 5. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Initialize Models
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Decision_Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive_Bayes": GaussianNB(),
        "Random_Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = []

    print("\n--- Starting Training Process ---")
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate AUC (Requires probabilities)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Capture metrics
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_proba),
            "Score": model.score(X_test, y_test),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1_Score": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred)
        }
        results.append(metrics)

        # Save individual model files
        joblib.dump(model, f'model/{name.lower()}_model.pkl')
        print(f"✅ Saved: model/{name.lower()}_model.pkl")

    # 7. Generate and Save Comparison Matrix
    matrix_df = pd.DataFrame(results)
    matrix_df.to_csv('model/comparison_matrix.csv', index=False)
    
    print("\n--- Training Complete ---")
    print(matrix_df)

if __name__ == "__main__":
    train_and_save_all_models()