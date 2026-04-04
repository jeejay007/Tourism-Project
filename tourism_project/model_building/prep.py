
import os
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
from google.colab import drive, userdata

# Define Project Paths
PROJECT_DIR = '/content/drive/MyDrive/Project10/tourism_project'
DATA_DIR = os.path.join(PROJECT_DIR, 'data/processed')
os.makedirs(DATA_DIR, exist_ok=True)

# 2. Load the dataset directly from Hugging Face
# Replace with your actual HF path
dataset_path = "GauthamJ007/VisitWithUs-Tourism-Dataset"
dataset = load_dataset(dataset_path)
df = dataset['train'].to_pandas()

# 3. Perform Data Cleaning
def clean_data(data):
    # Fix Gender inconsistency
    data['Gender'] = data['Gender'].replace('Fe Male', 'Female')

    # Standardize MaritalStatus
    data['MaritalStatus'] = data['MaritalStatus'].replace('Unmarried', 'Single')

    # Drop unnecessary identifier columns
    cols_to_drop = ['Unnamed: 0', 'CustomerID']
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

    # Fill missing values with median (Best practice: Median is robust to outliers)
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['MonthlyIncome'] = data['MonthlyIncome'].fillna(data['MonthlyIncome'].median())

    return data

df_cleaned = clean_data(df)

# 4. Split the cleaned dataset
train_df, test_df = train_test_split(
    df_cleaned,
    test_size=0.2,
    random_state=42,
    stratify=df_cleaned['ProdTaken']
)

# 5. Save locally to Google Drive (Persistent Storage)
# This allows you to inspect files without re-running the whole notebook
train_path = os.path.join(DATA_DIR, 'train.csv')
test_path = os.path.join(DATA_DIR, 'test.csv')

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Files saved locally to: {DATA_DIR}")

# 6. Upload resulting datasets back to Hugging Face (MLOps Pipeline Registry)
processed_dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})

# Push to Hub using Colab Secrets for the token
hf_token = userdata.get('HF_TOKEN')
processed_dataset.push_to_hub(
    "GauthamJ007/VisitWithUs-Tourism-Dataset-Processed",
    token=hf_token
)

print("Data cleaning, splitting, and HF registration completed.")
