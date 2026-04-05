import os
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# 1. Define Project Paths (Relative to the GitHub Repo Root)
# This replaces the Google Drive paths
DATA_DIR = 'tourism_project/data/processed'
os.makedirs(DATA_DIR, exist_ok=True)

# 2. Load the dataset directly from Hugging Face
# This ensures we are always pulling the latest registered version
dataset_path = "GauthamJ007/VisitWithUs-Tourism-Dataset"
dataset = load_dataset(dataset_path)

# Convert to pandas (Hugging Face datasets usually have a 'train' split by default)
if 'train' in dataset:
    df = dataset['train'].to_pandas()
else:
    # Fallback if no split is defined
    df = dataset.to_pandas()

# 3. Perform Data Cleaning
def clean_data(data):
    # Fix Gender inconsistency
    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].replace('Fe Male', 'Female')

    # Standardize MaritalStatus
    if 'MaritalStatus' in data.columns:
        data['MaritalStatus'] = data['MaritalStatus'].replace('Unmarried', 'Single')

    # Drop unnecessary identifier columns
    cols_to_drop = ['Unnamed: 0', 'CustomerID']
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

    # Fill missing values with median
    if 'Age' in data.columns:
        data['Age'] = data['Age'].fillna(data['Age'].median())
    if 'MonthlyIncome' in data.columns:
        data['MonthlyIncome'] = data['MonthlyIncome'].fillna(data['MonthlyIncome'].median())

    return data

df_cleaned = clean_data(df)

# 4. Split the cleaned dataset (Stratify ensures Prodtaken ratio stays same)
train_df, test_df = train_test_split(
    df_cleaned,
    test_size=0.2,
    random_state=42,
    stratify=df_cleaned['ProdTaken'] if 'ProdTaken' in df_cleaned.columns else None
)

# 5. Save locally within the GitHub Runner
# This allows the 'train.py' script in the next pipeline step to find these files
train_path = os.path.join(DATA_DIR, 'train.csv')
test_path = os.path.join(DATA_DIR, 'test.csv')

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Files saved locally within GitHub Runner to: {DATA_DIR}")

# 6. Upload resulting datasets back to Hugging Face
processed_dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})

# Retrieve the token from GitHub Secrets (passed as an environment variable in pipeline.yml)
hf_token = os.getenv('HF_TOKEN')

if hf_token:
    processed_dataset.push_to_hub(
        "GauthamJ007/VisitWithUs-Tourism-Dataset-Processed",
        token=hf_token
    )
    print("Cleaned data successfully pushed to Hugging Face!")
else:
    print("Error: HF_TOKEN not found in environment variables.Check your GitHub")
