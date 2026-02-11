
# Project Overview- 
In the rapidly evolving digital economy, identifying fraudulent transactions is critical for financial security. This dataset contains anonymized credit card transactions made by European cardholders in 2023.

Unlike older datasets, this version provides over 550,000 records, offering a more robust foundation for training high-accuracy machine learning models. The primary goal is to distinguish between legitimate transactions and fraudulent ones to prevent financial loss.

# Project usecase- 

we will be using 6 types of ML models for classification and then we will analyse these algorithym performance on the data as data is mostly clean so it be helpful for us to analyse algorithym performance with key below points comparison

# Machine learning models used -
1 Logistic Regression 
2 Decision Tree Classiﬁer
3 K-Nearest Neighbor Classiﬁer 
4 Naive Bayes Classiﬁer - Gaussian or Multinomial 
5 Ensemble Model - Random Forest 
6 Ensemble Model - XGBoost

# metrics for algorithym analysis- 
1 Accuracy
2 AUC Score (Area Under the Curve)
3 Precision
4 Recall
5 F1 Score
6 Matthews Correlation Coefficient (MCC Score)

# Dataset Characteristics-
The dataset is structured to be "ML-ready" while maintaining privacy through anonymization.
-Total Records: ~568,630 transactions
-Total Features: 31 (including ID and Class)
-Target Variable: Class (0 for Legitimate, 1 for Fraud)
-Data Format: CSV

# file structures- 

project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
    │-- .gitkeep (placeholder)
    │-- lr_model.pkl
    │-- dt_model.pkl



