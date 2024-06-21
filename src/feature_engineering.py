import pandas as pd

def add_total_income_feature(df):
    """
    Add Total Income feature by combining ApplicantIncome and CoapplicantIncome.
    """
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df.drop(columns=['ApplicantIncome', 'CoapplicantIncome'], inplace=True)
    return df

if __name__ == "__main__":
    train_data = pd.read_csv('../data/train_cleaned.csv')
    test_data = pd.read_csv('../data/test_cleaned.csv')
    train_data = add_total_income_feature(train_data)
    test_data = add_total_income_feature(test_data)
    train_data.to_csv('../data/train_featured.csv', index=False)
    test_data.to_csv('../data/test_featured.csv', index=False)
