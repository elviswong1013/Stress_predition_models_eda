from typing import Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np


def load_and_preprocess_data(filepath: str = "dataset.csv") -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, List[str], ColumnTransformer, pd.DataFrame, pd.DataFrame]:
    # Load data
    df = pd.read_csv(filepath)
    
    # Drop User_ID as it's an identifier
    if 'User_ID' in df.columns:
        df = df.drop('User_ID', axis=1)
    
    # Define features and target
    X = df.drop('Stress_Level', axis=1)
    y = df['Stress_Level']
    
    # Identify categorical and numerical columns
    categorical_cols = ['Gender', 'Occupation', 'Device_Type']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    # Create preprocessing pipeline
    # Standardize numerical features
    # One-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + list(cat_feature_names)
    
    return X_train_processed, X_test_processed, y_train, y_test, feature_names, preprocessor, X_train, X_test
