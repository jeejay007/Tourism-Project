from datasets import load_dataset, Dataset
import pandas as pd
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# 1. Load the data from the GITHUB REPO path (Not Drive!)
# Based on your structure: tourism_project/data/tourism.csv
data_path = "tourism_project/data/tourism.csv"
df = pd.read_csv(data_path)

# 2. Convert to Hugging Face Dataset format
hf_dataset = Dataset.from_pandas(df)

# 3. Push to the Hub using the secret token
# os.getenv("HF_TOKEN") pulls the secret from your pipeline.yml
token = os.getenv("HF_TOKEN")

hf_dataset.push_to_hub("GauthamJ007/VisitWithUs-Tourism-Dataset", token=token)

print("Data successfully registered on Hugging Face!")

repo_id = "GauthamJ007/VisitWithUs-Tourism-Dataset"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=token)

# Step 1: Check if the repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Repo '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=token)
    print(f"Repo '{repo_id}' created.")

# Step 2: Upload the folder from the GITHUB REPO path
# Based on your structure: tourism_project/data/
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
    token=token
)
