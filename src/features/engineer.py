import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.utils.logger import logger
from typing import Tuple

class FeatureEngineer:
    def __init__(self):
        self.preprocessor = None
        self.features = None
        
    def fit_transform(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Engineering features for training data...")
        
        # Separate target (Churn)
        y = train_df['Churn']
        X = train_df.drop(columns=['Churn', 'CustomerID'])
        
        self.features = X.columns
        
        # Identify column types
        numeric_features = X.select_dtypes(include=['int32', 'int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # Create transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Bundle transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names after one hot encoding
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = cat_encoder.get_feature_names_out(categorical_features)
        all_features = list(numeric_features) + list(cat_features)
        
        X_df = pd.DataFrame(X_processed, columns=all_features)
        return X_df, y
        
    def transform(self, test_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering features for testing data...")
        X = test_df.drop(columns=['CustomerID'])
        
        X_processed = self.preprocessor.transform(X)
        
        # Get feature names
        numeric_features = X.select_dtypes(include=['int32', 'int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = cat_encoder.get_feature_names_out(categorical_features)
        all_features = list(numeric_features) + list(cat_features)
        
        return pd.DataFrame(X_processed, columns=all_features)
