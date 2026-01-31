import pandas as pd
import numpy as np
import gradio as gr
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("diabetes.csv")

# Display first few rows
print(df.head())

# Display shape
print("Dataset shape:", df.shape)


# Replace 0 with NaN for invalid medical measurements
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Numerical preprocessing pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Apply preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, X.columns)
])


from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    pipeline, X_train, y_train, cv=5, scoring="accuracy"
)

print("CV Mean Accuracy:", cv_scores.mean())
print("CV Std Dev:", cv_scores.std())


from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__solver": ["liblinear", "lbfgs"]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

best_model = grid_search.best_estimator_

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = best_model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

with open("diabetes_pipeline.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Random Forest pipeline Saved as diabetes_pipeline.pkl")


# import numpy as np
# import pandas as pd
# import pickle

# # sklearn preprocessing

# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score

# df= pd.read_csv("diabetes.csv")
# df.head()

# print("Duplicate rows:", df.duplicated().sum())

# df = df.drop_duplicates()

# print("Shape After Removing Duplicates:", df.shape)

# # Target & Features
# X = df.drop('charges', axis=1)

# y = df["charges"]

# # Column Separation

# num_feat = X.select_dtypes(include=['int64', 'float64']).columns
# cat_feat = X.select_dtypes(include=['object']).columns


# num_transformer = Pipeline(
#     steps=[
#         ('imputer', SimpleImputer(strategy='median')),
#         ('scaler', StandardScaler())
#     ]
# )

# cat_transformer = Pipeline(
#     steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('encoder', OneHotEncoder(handle_unknown='ignore'))
#     ]
# )

# prepocessor = ColumnTransformer([
#     ('num', num_transformer, num_feat),
#     ('cat', cat_transformer, cat_feat)
# ])

# # Model

# rf_model = RandomForestRegressor(
#     n_estimators=200,
#     max_depth=10,
#     min_samples_split=2,
#     random_state=42,
#     n_jobs=-1
# )

# # Pipeline

# rf_pipeline = Pipeline(
#     steps=[
#         ('prepocessor', prepocessor),
#         ('model', rf_model)
#     ]
# )

# # train test split

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# cv_scores = cross_val_score(
#     rf_pipeline, X_train, y_train, cv=5, scoring="r2"
# )

# print(f"Mean r2:", cv_scores.mean())
# print("Standard Deviation:", cv_scores.std())

# param_grid = {
#     "model__n_estimators": [100, 200],
#     "model__max_depth": [10, 15, None],
#     "model__min_samples_split": [2, 5]
# }

# grid_search = GridSearchCV(
#     rf_pipeline, param_grid, cv=3, scoring="r2", n_jobs=-1
# )

# grid_search.fit(X_train, y_train)

# best_model = grid_search.best_estimator_

# y_pred = best_model.predict(X_test)

# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)

# print(f"RMSE: {rmse:.4f}")
# print(f"R2 Score: {r2:.4f}")
