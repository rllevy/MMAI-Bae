#Steps
# f1 score is 0.599
# Add in explicit numeric_features and categorical_features in case pycaret was misclassifying data
# Check if any categorical unique values exceed 25 (max for one hot encoding) -> none do
# Try a different imputation method (imputation = 'iterative')
# Imputation method drop -> f1 score 0.6999
# Try normalizing the data (set normalize=True, default is zscore) -> f1 score did not change

# Compare only a few models
# Try feature_selection = True, LightGBM error with JSON in column names, tried no preprocessing -> f1 score is 0.6135
# Try really long decision tree 

# Import necessary libraries
import pandas as pd
import re
from pycaret.classification import *



# Load the dataset
# file_path = "https://drive.google.com/uc?export=download&id=1eYCKuqJda4bpzXBVnqXylg0qQwvpUuum"
# path_to_test = 'https://drive.google.com/uc?export=download&id=1SmFBoNh7segI1Ky92mfeIe6TpscclMwQ'

file_path = '/Users/liamkopp/Downloads/mmai869.csv'
path_to_test = '/Users/liamkopp/Downloads/mmai869_competition.csv'

path_to_output = '/Users/liamkopp/Downloads/predictions.csv'

df = pd.read_csv(file_path)
df = df.replace('[^a-zA-Z0-9]', '', regex=True)
test_data = pd.read_csv(path_to_test)

numeric_features = [
          "h1n1_concern",
          "h1n1_knowledge",
          "behavioral_antiviral_meds",
          "behavioral_avoidance",
          "behavioral_face_mask",
          "behavioral_wash_hands",
          "behavioral_large_gatherings",
          "behavioral_outside_home",
          "behavioral_touch_face",
          "doctor_recc_h1n1",
          "doctor_recc_seasonal",
          "chronic_med_condition",
          "child_under_6_months",
          "health_worker",
          "health_insurance",
          "opinion_h1n1_vacc_effective",
          "opinion_h1n1_risk",
          "opinion_h1n1_sick_from_vacc",
          "opinion_seas_vacc_effective",
          "opinion_seas_risk",
          "opinion_seas_sick_from_vacc",
          "household_adults",
          "household_children",
]

categorical_features = [
    "age_group",
    "education",
    "race",
    "sex",
    "income_poverty",
    "marital_status",
    "rent_or_own",
    "employment_status",
    "hhs_geo_region",
    "census_msa",
    "employment_industry",
    "employment_occupation",
]


# Display the first few rows and info of the dataset
print(df.head())
print(df.info())

#0.7029 - removed outliers
#0.7065 - no remove outliers
#0.6992 - no fix imbalance
#0.7065 - normalize = True, default is z-score
#0.7132 - transformation = True, default is yeo-johnson, rare_to_vale = 0.01
#0.7096 - polynomial_features = True
#0.6818 - feature_select=True (classic = SelectFromModel)
#0.7048 - remove_multicollinearity (default 0.9 pearson correlation)
#0.6683 - PCA (default is SVD and  0.99 target percentage for information retention)
#0.7125 - Took away rare_to_value (minimum fraction of category occurrences in a column)
# - iterative imputation using LightGBM (way too long)
#0.68... - XGBoost
#
# Initialize the setup
clf1 = setup(
    data=df, 
    # imputation_type='iterative',
    imputation_type='simple',
    numeric_imputation='drop',
    categorical_imputation='drop', 
    fix_imbalance=True,
    # normalize=True,
    transformation=True,
    # polynomial_features=True,
    # feature_selection=True,
    # remove_multicollinearity=True,
    # pca=True,
    rare_to_value=0.01, 
    target='h1n1_vaccine', 
    session_id=123, 
    # log_experiment=True, 
    # experiment_name='h1n1_vaccine_classification',
    system_log=False,
    categorical_features=categorical_features, 
    numeric_features=numeric_features, )

# Compare models and select the top 5 models
# best_models = compare_models(sort='F1')
# best_models = compare_models(sort='F1', include=['lr','knn','nb','dt','svm','rbfsvm','gpc','mlp','ridge','rf','qda','ada','gbc','lda','et','xgboost','lightgbm','catboost'])

gbc = create_model('catboost')
tuned = tune_model(gbc, optimize = 'F1')
# print(best_models)

# Predict labels for the test dataset
# predictions = predict_model(best_models, data=test_data)

# # Output csv of predictions
# predictions['prediction_label'].to_csv(path_to_output, index=False)




# Proof of no duplicate rows (instances)
# print(len(df))
# newdf = df.drop_duplicates()
# print(len(newdf))