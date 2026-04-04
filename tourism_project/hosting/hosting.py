
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# Replace with your Space name and Username
SPACE_ID = "GauthamJ007/VisitWithUs-Wellness-Tourism-Project-Space"

# Create the Space if it doesn't exist (SDK/Docker type)
api.create_repo(repo_id=SPACE_ID, repo_type="space", space_sdk="docker", exist_ok=True)

# Define Google Drive deployment path
folder_path="/content/drive/MyDrive/Project10/tourism_project/deployment"

files_to_upload = ["app.py", "requirements.txt", "Dockerfile"]

for file in files_to_upload:
    api.upload_file(
        path_or_fileobj=os.path.join(folder_path, file),
        path_in_repo=file,
        repo_id=SPACE_ID,
        repo_type="space"
    )

print(f"Deployment successful! Access your API at: https://huggingface.co/spaces/{SPACE_ID}")
