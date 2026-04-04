
import pandas as pd
import joblib
import os
from datasets import load_dataset
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from huggingface_hub import HfApi

# 1. Load data
dataset = load_dataset("GauthamJ007/VisitWithUs-Tourism-Dataset-Processed")
train_df = dataset['train'].to_pandas()
test_df = dataset['test'].to_pandas()

X_train = train_df.drop(columns=['ProdTaken'])
y_train = train_df['ProdTaken']
X_test = test_df.drop(columns=['ProdTaken'])
y_test = test_df['ProdTaken']

# --- MLOPS ADAPTATION: Define Preprocessing ---
# Identifying numeric and categorical columns for the pipeline
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ]
)

# 2. Define the Full Pipeline (Preprocessor + Classifier)
# This code handles encoding automatically
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# 3. Tune the Model (Grid Search)
# Note the prefix 'classifier__' to reach parameters inside the pipeline
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.1],
    'classifier__subsample': [0.8]
}

print("Starting Hyperparameter Tuning with Pipeline...")
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    verbose=1
)

# We fit on the RAW X_train (no get_dummies needed!)
grid_search.fit(X_train, y_train)

# 4. Evaluate the Best Pipeline
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test)
print("\nModel Evaluation Report:")
print(classification_report(y_test, y_pred))

# 5. Register the FULL PIPELINE in Hugging Face
# This saves the OneHotEncoder AND the XGBoost model together
model_name = "VisitWithUs-Wellness-Tourism-Predictor"
model_dir = "best_model_dir"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "xgboost_model.joblib")

# CRITICAL: We save 'best_pipeline', not just the model
joblib.dump(best_pipeline, model_path)

# Push to Hugging Face
api = HfApi()
repo_id = f"GauthamJ007/{model_name}"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

api.upload_folder(
    folder_path=model_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Registering best Pipeline (Encoder + XGBoost)"
)

print(f"Full Pipeline successfully registered at: huggingface.co/{repo_id}")
