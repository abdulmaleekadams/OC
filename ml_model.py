import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Load and preprocess the dataset
cancer_data = pd.read_csv('ovarian.csv')

# ... (data preprocessing code)
cancer_data = cancer_data.apply(
    lambda x: x.str.rstrip() if x.dtype == "object" else x)
cancer_data.replace({'AFP': {'>1210.00': '1210.00', '>1210': '1210.00'},
                    'CA125': {'>5000.00': '5000.00'},
                     'CA19-9': {'>1000.00': '1000.00', '>1000': '1000.00', '<0.600': '0.5'}}, inplace=True)

# Convert object columns to float columns
for col in cancer_data.drop('TYPE', axis=1).select_dtypes(include=['object']).columns:
    cancer_data[col] = cancer_data[col].astype('float')

# Convert target column to integer
cancer_data['TYPE'] = cancer_data['TYPE'].astype('int')

# cols_to_drop = ['CA72-4', 'NEU']
cols_to_drop = ['CA72-4', 'SUBJECT_ID']
cancer_data = cancer_data.drop(cols_to_drop, axis=1)

# get columns with missing data
cols_with_missing = [
    col for col in cancer_data.columns if cancer_data[col].isnull().any()]

# impute missing data with median value
for col in cols_with_missing:
    median_val = cancer_data[col].median()
    cancer_data[col].fillna(median_val, inplace=True)

# Split data into features (X) and target (y)
cancer_X_train = cancer_data.drop('TYPE', axis=1)
cancer_y_train = cancer_data['TYPE']

selected_features = ['Age', 'CEA', 'IBIL', 'NEU', 'Menopause', 'CA125', 'ALB', 'HE4', 'GLO', 'LYM%']

# Split data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(
    cancer_X_train, cancer_y_train, test_size=0.3, random_state=42)
X_train = X_train[selected_features]
X_test = X_test[selected_features]

# Train the model if not already trained
try:
    xgb_model = joblib.load("xgb_model.joblib")
except FileNotFoundError:
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, "xgb_model.joblib")
