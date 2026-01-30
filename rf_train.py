import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =====================
# Load dataset
# =====================
df = pd.read_csv("Loan prediction.csv")  

print(df.head())

# Drop ID column if exists
if 'Loan_ID' in df.columns:
    df.drop(columns=['Loan_ID'], inplace=True)

# =====================
# Target and features
# =====================
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']   # Y / N

# =====================
# Column split
# =====================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# =====================
# Preprocessing
# =====================
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])

# =====================
# Random Forest Classifier
# =====================
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# =====================
# Full Pipeline
# =====================
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])

# =====================
# Train-test split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================
# Train
# =====================
rf_pipeline.fit(X_train, y_train)

# =====================
# Evaluation
# =====================
y_pred = rf_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Y')
recall = recall_score(y_test, y_pred, pos_label='Y')
f1 = f1_score(y_test, y_pred, pos_label='Y')

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# =====================
# Save model
# =====================
with open("loan_rf_pipeline.pkl", "wb") as f:
    pickle.dump(rf_pipeline, f)

print("✅ Random Forest classification pipeline saved as loan_rf_pipeline.pkl")
