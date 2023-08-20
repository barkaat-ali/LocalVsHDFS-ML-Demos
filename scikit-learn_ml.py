import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv("1.csv")
data["bmi"] = pd.to_numeric(data["bmi"], errors="coerce")

# Convert categorical variables to numerical using LabelEncoder
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
label_encoders = {col: LabelEncoder().fit(data[col].astype(str)) for col in categorical_cols}
for col, encoder in label_encoders.items():
    data[col + "_index"] = encoder.transform(data[col].astype(str))

# Impute null values with the mean of the "bmi" column
imputer = SimpleImputer(strategy="mean")
data["bmi_imputed"] = imputer.fit_transform(data[["bmi"]])

# Prepare feature columns and target variable
feature_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi_imputed"] + [col + "_index" for col in categorical_cols]
X = data[feature_cols]
y = data["stroke"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
predictions = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

