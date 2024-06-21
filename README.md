# Loan Eligibility Prediction

This project aims to predict loan eligibility based on various customer features using machine learning techniques. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and prediction.


This is an online application form to check your eligibility for loan approval. The application uses a deployed Streamlit app.

## Deployed Application

You can access the deployed application to check your loan eligibility using the following link:

[Loan Eligibility Prediction Application](https://loan-eligibility-prediction-capabl-ml-hackathon.streamlit.app/)

## Usage Instructions

1. Click on the link above to open the application.
2. Fill out the required fields in the form.
3. Submit the form to check your loan eligibility.
4. The application will provide you with an instant decision on your loan eligibility based on the input provided.

## Application Form Fields

The form requires the following information:
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Gender
- Marital Status
- Education
- Self Employed Status
- Property Area
- Dependents

## Additional Information

This application is built using Streamlit, a popular framework for creating data apps in Python. The source code for the application can be found in the GitHub repository linked above.


## Project Structure

The project directory is structured as follows:



```bash
loan-eligibility-prediction/
│
├── data/
│ ├── loan_predictions.csv # Predictions file for test data
│ ├── model_training.csv # Final training data used for model training
│ ├── test_cleaned.csv # Cleaned and preprocessed test data
│ ├── test.csv # Original test data
│ ├── train_cleaned.csv # Cleaned and preprocessed training data
│ └── train.csv # Original training data
│
├── model/
│ ├── label_encoders.pkl # Saved LabelEncoders for categorical variables
│ ├── random_forest_model.pkl # Trained Random Forest model
│ ├── random_forest_model(1).pkl # Additional trained Random Forest model
│ ├── scaler.pkl # Saved StandardScaler for feature scaling
│ └── status_encoder.pkl # Saved LabelEncoder for target variable
│
├── notebooks/
│ ├── EDA.ipynb # Exploratory Data Analysis notebook
│ └── ModelBuilding.ipynb # Model building and evaluation notebook
│
├── src/
│ ├── data_preprocessing.py # Custom functions for data loading and preprocessing
│ ├── feature_engineering.py # Functions for feature engineering
│ ├── model_training.py # Functions for model training and evaluation
│ └── predictions.py # Functions for making predictions and saving results
│
├── app.py # Streamlit deployment app run file
├── requirements.txt # Python dependencies
└── README.md # Project overview and instructions
```

### Explanation
- Each directory (`data/`, `model/`, `notebooks/`, `src/`) and their respective files are listed with comments indicating their purpose.
- Use of indentation and symbols (`├──` for directories and `│`, `└──` for files) to visually represent the structure.
- Ensure each file and directory name matches exactly as per your project structure.

## Components

### 1. Exploratory Data Analysis (EDA)

- **EDA.ipynb**: This notebook performs data loading, cleaning, and exploratory analysis on the `train.csv` and `test.csv` datasets. It generates `train_cleaned.csv` and `test_cleaned.csv` after handling missing values and basic statistical analysis. Visualizations include histograms, heatmaps for missing values, correlation analysis, and categorical vs target variable analysis.

### 2. Model Building and Prediction

- **ModelBuilding.ipynb**: Loads preprocessed data (`train_cleaned.csv` and `test_cleaned.csv`), combines them for consistent encoding, adds a Total Income feature, encodes categorical variables, trains a Random Forest Classifier, evaluates its performance, and saves the model and encoders:

- `random_forest_model.pkl`: Trained Random Forest model
- `random_forest_model(1).pkl`: Additional trained Random Forest model
- `label_encoders.pkl`: Saved LabelEncoders for categorical variables
- `status_encoder.pkl`: Saved LabelEncoder for target variable
- `scaler.pkl`: Saved StandardScaler for feature scaling

Finally, it makes predictions on the test set and saves results in `loan_predictions.csv`.

## Data Merging

After obtaining predictions, merge them with the cleaned test data to create the final training dataset:

```python
import pandas as pd

# Load the CSV files for merging
loan_predictions = pd.read_csv('../data/loan_predictions.csv')
test_cleaned = pd.read_csv('../data/test_cleaned.csv')

# Drop the 'Loan_ID' column from loan_predictions
loan_predictions = loan_predictions.drop(columns=['Loan_ID'])

# Add remaining columns to test_cleaned
merged_data = pd.concat([test_cleaned, loan_predictions], axis=1)

# Save the merged data to model_training.csv
merged_data.to_csv('../data/model_training.csv', index=False)

print("Merged data saved as model_training.csv")

```

### 3. Source Code (`src/`)

- **data_preprocessing.py**: Defines functions for loading data (`load_data`) and handling missing values (`handle_missing_values`).

- **feature_engineering.py**: Contains `add_total_income_feature` function to add a new feature `Total_Income`.

- **model_training.py**: Provides functions for encoding categorical variables (`encode_categorical_variables`) and training the Random Forest model (`train_model`).

- **predictions.py**: Implements functions for making predictions (`make_predictions`) using the trained model and saving predictions (`save_predictions`).

### 4. Requirements

- **requirements.txt**: Lists Python dependencies required to run the project.

## Usage

1. **Setup Environment**: Install dependencies listed in `requirements.txt`.

   ```bash
   pip install -r requirements.txt

2. **Exploratory Data Analysis**: Run `EDA.ipynb` to explore and preprocess the data.

3. **Model Building**: Execute `ModelBuilding.ipynb` to train the Random Forest model and generate predictions.

4. **Streamlit Deployment**: Run the app.py file to start the Streamlit app for the online application form.

   ```bash
   streamlit run app.py
   ```

6. **Customization**: Modify scripts in `src/` for specific data preprocessing, feature engineering, or model training needs.

7. **Documentation**: Modify `README.md` to reflect any changes, including detailed explanations of each script's purpose and usage.

### Notes
- Ensure data paths (`../data/`) are correctly configured in notebooks and scripts.
- Customize preprocessing, feature engineering, or model parameters based on specific project requirements.

## Team Members

1. **Chittala Pavan Kalyan**
   - **Email**: pavankalyanchittala0@gmail.com
   - **Capabl ID**: CPBLaak0485

2. **Gandham Dev Amarnadh**
   - **Email**: amarndhgandham000@gmail.com
   - **Capabl ID**: CPBLaak0486
