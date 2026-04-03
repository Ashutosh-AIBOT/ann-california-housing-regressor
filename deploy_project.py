import os
from huggingface_hub import HfApi

# --- CONFIGURATION ---
GITHUB_USER = "Ashutosh-AIBOT"
HF_USER = "Ashutosh1975"
REPO_NAME = "ann-california-housing-regressor"
PROJECT_FOLDER = "/home/ashutosh/Desktop/NO-AI-USE/Deep-learning/ANN Regression"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

if not GITHUB_TOKEN or not HF_TOKEN:
    print("⚠️ WARNING: GITHUB_TOKEN or HF_TOKEN not found in environment variables.")
    print("Deployment might fail without these credentials.")

api_hf = HfApi(token=HF_TOKEN)

def create_github_repo(token, name):
    url = "https://api.github.com/user/repos"
    headers = {"Authorization": f"token {token}"}
    data = {
        "name": name,
        "description": "Professional California Housing Price Regressor built with PyTorch and deployed as a Dockerized Streamlit dashboard.",
        "auto_init": False,
        "private": False
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        print(f"✅ Created GitHub repo: {name}")
    elif response.status_code == 422:
        print(f"ℹ️ GitHub repo {name} already exists.")
    else:
        print(f"❌ GitHub Error: {response.status_code} - {response.text}")

def create_hf_space(token, id):
    try:
        api_hf.create_repo(repo_id=id, repo_type="space", space_sdk="docker", private=False)
        print(f"✅ Created/Verified HF Space: {id}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"ℹ️ HF Space {id} already exists.")
        else:
            print(f"❌ HF Error: {e}")

def upload_to_hf(folder, id):
    print(f"🚀 Uploading to Hugging Face: {id}...")
    api_hf.upload_folder(
        folder_path=folder,
        repo_id=id,
        repo_type="space",
        allow_patterns=["*.py", "*.txt", "*.md", "Dockerfile", "models/*", "data/artifacts/*", "data/raw/*", "charts/*", "data/processed/*.pkl"],
        ignore_patterns=["notebooks/*", "__pycache__/*", "*.pdf", "*.npz", ".git/*", ".gitattributes", "deploy_project.py"],
    )
    print(f"✅ HF Upload Complete!")

if __name__ == "__main__":
    # 1. Create Cloud Repos
    create_github_repo(GITHUB_TOKEN, REPO_NAME)
    create_hf_space(HF_TOKEN, f"{HF_USER}/{REPO_NAME}")

    # 2. Upload to HF
    upload_to_hf(PROJECT_FOLDER, f"{HF_USER}/{REPO_NAME}")

    print("-" * 30)
    print(f"🚀 GITHUB: https://github.com/{GITHUB_USER}/{REPO_NAME}")
    print(f"🚀 HF SPACE: https://huggingface.co/spaces/{HF_USER}/{REPO_NAME}")
    print("-" * 30)
