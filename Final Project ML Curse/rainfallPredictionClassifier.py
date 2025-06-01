import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns # Seaborn was imported but not used in the provided snippet, kept for potential use

# Load the dataset
df = pd.read_csv(r"/home/zacc-marc/Desktop/Zacc Workspace/Machine Learning Project/Final Project ML Curse/data/weatherAUS.csv")

# --- Data Preprocessing ---

# Drop rows with any missing values (Note: This can significantly reduce dataset size; imputation is often preferred)
df = df.dropna()

# Rename columns: 'RainTomorrow' becomes the target 'RainToday'.
# 'RainToday' (original) becomes 'RainYesterday', a predictor.
df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })

# Filter for specific locations
df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]

# Convert 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date']) # Completed

# Feature Engineering: Create 'Season' from 'Date'
def date_to_season(date_obj): # Renamed 'date' to 'date_obj' to avoid conflict with 'Date' column name
    month = date_obj.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'

df['Season'] = df['Date'].apply(date_to_season)

# Define columns to potentially drop
columns_to_drop = ['Date', 'RISK_MM']

# Create a list of columns that actually exist in the DataFrame to avoid errors
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]

if existing_columns_to_drop:
    df = df.drop(columns=existing_columns_to_drop)
    print(f"Dropped columns: {existing_columns_to_drop}")
else:
    print("No columns from the specified list (Date, RISK_MM) found to drop.")

# print(df.head()) # Optional: to check df after drops

# Define features (X) and target (y)
# 'RainToday' (which was originally 'RainTomorrow') is the target.
X = df.drop(columns='RainToday', axis=1) # Completed
y = df['RainToday'] # Completed

# --- Exploratory Data Analysis (Exercises) ---

"""
Exercise 3. How balanced are the classes?
Display the counts of each class.
"""
print("\nExercise 3: Class distribution for 'RainToday'")
print(y.value_counts()) # Completed

"""
Exercise 4. What can you conclude from these counts?
How often does it rain annually in the Melbourne area?
How accurate would you be if you just assumed it won't rain every day?
Is this a balanced dataset?
Next steps?

* To determine how often it rains annually, you'd analyze the proportion of 'Yes' in 'RainToday' from the value_counts.
* If you assumed it won't rain every day, your accuracy would be the proportion of 'No' in 'RainToday'.
* The dataset is likely imbalanced if one class significantly outnumbers the other (common for rain prediction).
* Next steps often involve addressing imbalance (e.g., using class_weight, oversampling, undersampling) 
    and proceeding with model training and evaluation, paying attention to metrics like recall and F1-score for the minority class.
"""

# --- Data Splitting ---

"""
Exercise 5. Split data into training and test sets, ensuring target stratification
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42) # Completed
print(f"\nTraining set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Test set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

# --- Feature Preprocessing Setup ---

# Identify numeric and categorical features from the training set
numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()  # Completed (using 'number' is more general)
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist() # Completed

print(f"\nNumeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# Create preprocessing pipelines for numeric and categorical features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features), # Completed
        ('cat', categorical_transformer, categorical_features) # Completed
    ]
)

# --- RandomForestClassifier Model Training ---
print("\n--- Training RandomForestClassifier ---")

# Create the full pipeline with preprocessor and classifier
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), # Completed
    ('classifier', RandomForestClassifier(random_state=42)) # Completed (named step 'classifier')
])

# Define parameter grid for RandomForest
rf_param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Added random_state for reproducibility

# Setup GridSearchCV for RandomForest
rf_grid_search = GridSearchCV(estimator=rf_pipeline, param_grid=rf_param_grid, cv=cv, scoring='accuracy', verbose=1)  # Completed estimator, cv; reduced verbosity
rf_grid_search.fit(X_train, y_train) # Completed

print("\nBest parameters found for RandomForest: ", rf_grid_search.best_params_)
print("Best cross-validation score for RandomForest: {:.2f}".format(rf_grid_search.best_score_))

# Evaluate RandomForest on the test set
rf_test_score = rf_grid_search.score(X_test, y_test) # Completed
print("RandomForest Test set score: {:.2f}".format(rf_test_score))

# Make predictions with the best RandomForest model
y_pred_rf = rf_grid_search.predict(X_test) # Completed

print("\nRandomForest Classification Report:")
print(classification_report(y_test, y_pred_rf)) # Completed

# Display RandomForest Confusion Matrix
rf_conf_matrix = confusion_matrix(y_test, y_pred_rf) # Completed
rf_disp = ConfusionMatrixDisplay(confusion_matrix=rf_conf_matrix, display_labels=rf_grid_search.classes_) # Completed, added display_labels
rf_disp.plot(cmap='Blues')
plt.title('RandomForest Confusion Matrix')
plt.show()

# --- RandomForest Feature Importances ---
print("\n--- RandomForest Feature Importances ---")

# Get feature importances from the best RandomForest model
# The preprocessor step for categorical features creates new column names after one-hot encoding.
# We need to get these transformed names.
best_rf_model = rf_grid_search.best_estimator_
transformed_categorical_features = list(best_rf_model.named_steps['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))
feature_names_rf = numeric_features + transformed_categorical_features

rf_importances = best_rf_model.named_steps['classifier'].feature_importances_

importance_df_rf = pd.DataFrame({'Feature': feature_names_rf,
                                 'Importance': rf_importances
                                }).sort_values(by='Importance', ascending=False)

N = 20  # Number of top features to display
top_features_rf = importance_df_rf.head(N)

# Plotting top N features for RandomForest
plt.figure(figsize=(10, 8)) # Adjusted figure size for better readability
plt.barh(top_features_rf['Feature'], top_features_rf['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title(f'Top {N} Most Important Features (RandomForest)')
plt.xlabel('Importance Score')
plt.tight_layout() # Added for better spacing
plt.show()

# --- LogisticRegression Model Training ---
print("\n--- Training LogisticRegression ---")

# Create a new pipeline for Logistic Regression (or modify the existing one if careful)
# It's often cleaner to define a new pipeline object if you plan to compare them later.
# Here, we'll modify the existing 'rf_pipeline' structure for Logistic Regression.
# Note: The 'pipeline' variable from the original snippet was rf_pipeline.
# We are essentially creating a new pipeline configuration for Logistic Regression.

logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), # Uses the same preprocessor
    ('classifier', LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)) # Set solver and max_iter
])

# Define parameter grid for LogisticRegression
logreg_param_grid = {
    'classifier__penalty': ['l1', 'l2'],
    'classifier__C': [0.01, 0.1, 1, 10, 100], # Common C values to test regularization strength
    'classifier__class_weight' : [None, 'balanced']
}
# Note: 'solver' is often part of the grid if testing multiple solvers, 
# but here it's fixed to 'liblinear' which supports l1 and l2.

# Setup GridSearchCV for LogisticRegression
logreg_grid_search = GridSearchCV(estimator=logreg_pipeline, param_grid=logreg_param_grid, cv=cv, scoring='accuracy', verbose=1) # Reduced verbosity
logreg_grid_search.fit(X_train, y_train) # Completed

print("\nBest parameters found for LogisticRegression: ", logreg_grid_search.best_params_)
print("Best cross-validation score for LogisticRegression: {:.2f}".format(logreg_grid_search.best_score_))

# Evaluate LogisticRegression on the test set
logreg_test_score = logreg_grid_search.score(X_test, y_test)
print("LogisticRegression Test set score: {:.2f}".format(logreg_test_score))

# Make predictions with the best LogisticRegression model
y_pred_logreg = logreg_grid_search.predict(X_test) # Completed

print("\nLogisticRegression Classification Report:")
print(classification_report(y_test, y_pred_logreg))

# Display LogisticRegression Confusion Matrix
logreg_conf_matrix = confusion_matrix(y_test, y_pred_logreg)
logreg_disp = ConfusionMatrixDisplay(confusion_matrix=logreg_conf_matrix, display_labels=logreg_grid_search.classes_)
logreg_disp.plot(cmap='Greens') # Changed cmap for visual distinction
plt.title('LogisticRegression Confusion Matrix')
plt.show()

# --- Optional: Logistic Regression Coefficients (Feature Importances) ---
# For Logistic Regression, coefficients can be inspected instead of 'feature_importances_'
# This requires the best_estimator_ and handling of one-hot encoded feature names similar to RF.
print("\n--- LogisticRegression Coefficients ---")
best_logreg_model = logreg_grid_search.best_estimator_

# Ensure the classifier step is indeed LogisticRegression
if hasattr(best_logreg_model.named_steps['classifier'], 'coef_'):
    logreg_coeffs = best_logreg_model.named_steps['classifier'].coef_[0] # Coeffs are often in a 2D array

    # Feature names are the same as for RF (numeric_features + transformed_categorical_features)
    # feature_names_logreg = feature_names_rf (already defined)

    coeffs_df = pd.DataFrame({'Feature': feature_names_rf, 
                              'Coefficient': logreg_coeffs
                             }).sort_values(by='Coefficient', key=abs, ascending=False) # Sort by absolute value

    top_coeffs_logreg = coeffs_df.head(N)

    plt.figure(figsize=(10, 8))
    plt.barh(top_coeffs_logreg['Feature'], top_coeffs_logreg['Coefficient'], color='lightcoral')
    plt.gca().invert_yaxis()
    plt.title(f'Top {N} Features by Coefficient Magnitude (LogisticRegression)')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.show()
else:
    print("Could not retrieve coefficients, classifier might not be Logistic Regression or might not have 'coef_'.")

print("\n--- End of Script ---")