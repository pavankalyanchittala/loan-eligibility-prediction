import pandas as pd
import joblib

def make_predictions(model, test_data):
    predictions = model.predict(test_data)
    return predictions

def save_predictions(predictions, test_data, filename='../data/loan_predictions.csv'):
    submission = pd.DataFrame({'Loan_ID': test_data['Loan_ID'], 'Loan_Status': predictions})
    submission.to_csv(filename, index=False)

if __name__ == "__main__":
    test_data = pd.read_csv('../data/test_featured.csv')
    model = joblib.load('../model/random_forest_model.pkl')
    
    predictions = make_predictions(model, test_data)
    save_predictions(predictions, test_data)
