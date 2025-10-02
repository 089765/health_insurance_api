import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def preprocess_data(df):
    """
    Convert categorical string columns to numeric using one-hot encoding.
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded

def train_model_from_csv(csv_path, model_path="models/model.joblib", target_column="charges"):
    """
    Train a Linear Regression model from a CSV file and save the model.
    """
    df = pd.read_csv(csv_path)
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in dataset")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X = preprocess_data(X)  # Encode categorical columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    return {"message": "Model trained successfully", "features": list(X.columns)}

def test_model_from_csv(csv_path, model_path="models/model.joblib", target_column="charges"):
    """
    Evaluate the trained model on a test CSV file and return metrics.
    """
    df = pd.read_csv(csv_path)
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in dataset")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X = preprocess_data(X)  # Encode categorical columns

    model = joblib.load(model_path)
    
    # Ensure test data has same features as training
    for col in model.feature_names_in_:
        if col not in X.columns:
            X[col] = 0  # Add missing dummy columns
    X = X[model.feature_names_in_]  # Reorder columns

    y_pred = model.predict(X)

    return {
        "mae": mean_absolute_error(y, y_pred),
        "mse": mean_squared_error(y, y_pred),
        "r2": r2_score(y, y_pred)
    }

def predict_from_dict(data_dict, model_path="models/model.joblib"):
    """
    Make a prediction from a single input dictionary.
    """
    model = joblib.load(model_path)
    df = pd.DataFrame([data_dict])
    df = preprocess_data(df)  # Encode categorical columns
    
    # Ensure all features used in training are present
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    df = df[model.feature_names_in_]  # Reorder columns

    y_pred = model.predict(df)[0]
    return {"prediction": float(y_pred)}
