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
from huggingface_hub.utils import RepositoryNotFoundError

# --- 1. CONFIG ---
DATASET_ID = "GauthamJ007/VisitWithUs-Tourism-Dataset-Processed"
MODEL_NAME = "VisitWithUs-Wellness-Tourism-Predictor"
MODEL_FILENAME = "tourism_pipeline.joblib"

print("🚀 Loading dataset...")
dataset = load_dataset(DATASET_ID)

train_df = dataset['train'].to_pandas()
test_df = dataset['test'].to_pandas()

X_train = train_df.drop(columns=['ProdTaken'])
y_train = train_df['ProdTaken']
X_test = test_df.drop(columns=['ProdTaken'])
y_test = test_df['ProdTaken']

# --- 2. FEATURE TYPES ---
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

print("Numeric Features:", numeric_features)
print("Categorical Features:", categorical_features)

# --- 3. PREPROCESSOR ---
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# --- 4. MODEL ---
# Handle class imbalance
class_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric='logloss'
)

# --- 5. FULL PIPELINE ---
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# --- 6. HYPERPARAMETER TUNING (Balanced Grid) ---
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__subsample': [0.8]
}

print("⚙️ Starting Hyperparameter Tuning...")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# --- 7. BEST MODEL ---
best_pipeline = grid_search.best_estimator_

print("✅ Best Parameters:", grid_search.best_params_)

# --- 8. EVALUATION ---
y_pred = best_pipeline.predict(X_test)

print("\n📊 Model Evaluation Report:")
print(classification_report(y_test, y_pred))

# --- 9. SAVE MODEL ---
model_dir = "artifacts"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, MODEL_FILENAME)

# Save FULL pipeline (preprocessing + model)
joblib.dump(best_pipeline, model_path)

print(f"📦 Model saved at: {model_path}")

# --- 10. PUSH TO HUGGING FACE ---
api = HfApi()
repo_id = f"GauthamJ007/{MODEL_NAME}"

try:
    api.repo_info(repo_id=repo_id, repo_type="model")
    print(f"📁 Repo exists: {repo_id}")
except RepositoryNotFoundError:
    print(f"📁 Creating repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", private=False)

# Upload model
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=MODEL_FILENAME,
    repo_id=repo_id,
    repo_type="model"
)

print(f"🚀 Model successfully uploaded to: https://huggingface.co/{repo_id}")
