import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def encode_categorical_variables(train_data, test_data):
    """
    Encode categorical variables using LabelEncoder.
    """
    label_encoders = {}
    for column in train_data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        train_data[column] = label_encoders[column].fit_transform(train_data[column])
        test_data[column] = label_encoders[column].transform(test_data[column])
    return train_data, test_data

def train_model(X_train, y_train, X_val, y_val):
    """
    Train the Random Forest Classifier.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    confusion = confusion_matrix(y_val, y_pred)
    classification_rep = classification_report(y_val, y_pred)
    
    return model, accuracy, confusion, classification_rep

if __name__ == "__main__":
    train_data = pd.read_csv('../data/train_featured.csv')
    test_data = pd.read_csv('../data/test_featured.csv')
    train_data, test_data = encode_categorical_variables(train_data, test_data)
    
    X = train_data.drop(columns=['Loan_Status'])
    y = train_data['Loan_Status']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model, accuracy, confusion, classification_rep = train_model(X_train, y_train, X_val, y_val)
    
    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{confusion}')
    print(f'Classification Report:\n{classification_rep}')
    
    # Save the model
    import joblib
    joblib.dump(model, '../model/random_forest_model(1).pkl')
