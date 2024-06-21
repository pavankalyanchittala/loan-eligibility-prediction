import pandas as pd

def load_data(train_path, test_path):
    """
    Load train and test datasets.
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    """
    # Fill missing values with the median for numerical columns
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Fill missing values with the mode for categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df
