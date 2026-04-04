
from datasets import load_dataset, Dataset
import pandas as pd
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# 1. Load the local data
df = pd.read_csv("/content/drive/MyDrive/Project10/tourism_project/data/tourism.csv")

# 2. Convert to Hugging Face Dataset format
hf_dataset = Dataset.from_pandas(df)

# 3. Push to the Hub (Requires 'huggingface-cli login' or a token)
# Replace 'your-username' with your actual Hugging Face username
hf_dataset.push_to_hub("GauthamJ007/VisitWithUs-Tourism-Dataset")

print("Data successfully registered on Hugging Face!")



repo_id = "GauthamJ007/VisitWithUs-Tourism-Dataset"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="/content/drive/MyDrive/Project10/tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
