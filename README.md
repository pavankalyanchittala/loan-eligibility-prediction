# Loan Eligibility Prediction

This project aims to predict loan eligibility based on various customer features using machine learning techniques. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and prediction.

## Project Structure

The project directory is structured as follows:
loan-eligibility-prediction/
│
├── data/
│ ├── train.csv # Original training data
│ ├── test.csv # Original test data
│ ├── train_cleaned.csv # Cleaned and preprocessed training data
│ ├── test_cleaned.csv # Cleaned and preprocessed test data
│ └── loan_predictions.csv # Predictions file for test data
│
├── model/
│ └── random_forest_model.pkl # Trained Random Forest model
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
├── requirements.txt # Python dependencies
└── README.md # Project overview and instructions


## Components

### 1. Exploratory Data Analysis (EDA)

- **EDA.ipynb**: This notebook performs data loading, cleaning, and exploratory analysis on the `train.csv` and `test.csv` datasets. It generates `train_cleaned.csv` and `test_cleaned.csv` after handling missing values and basic statistical analysis. Visualizations include histograms, heatmaps for missing values, correlation analysis, and categorical vs target variable analysis.

### 2. Model Building and Prediction

- **ModelBuilding.ipynb**: Loads preprocessed data (`train_cleaned.csv` and `test_cleaned.csv`), combines them for consistent encoding, adds a `Total Income` feature, encodes categorical variables, trains a Random Forest Classifier, evaluates its performance, and saves the model as `random_forest_model.pkl`. Finally, it makes predictions on the test set and saves results in `loan_predictions.csv`.

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

4. **Customization**: Modify scripts in `src/` for specific data preprocessing, feature engineering, or model training needs.

5. **Documentation**: Modify `README.md` to reflect any changes, including detailed explanations of each script's purpose and usage.

### Notes
- Ensure data paths (`../data/`) are correctly configured in notebooks and scripts.
- Customize preprocessing, feature engineering, or model parameters based on specific project requirements.

## Team Members

1. **Chittala Pavan Kalyan**
   - **Email**: pavankalyanchittala0@gmail.com
   - **Capabl ID**: CPBLaak0485

2. **Dev Amarnadh Gandham**
   - **Email**: amarndhgandham000@gmail.com
   - **Capabl ID**: CPBLaak0486
