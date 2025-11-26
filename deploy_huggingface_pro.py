import os
import sys
import subprocess

# Ensure huggingface_hub is installed
try:
    import huggingface_hub
except ImportError:
    print("Installing huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    import huggingface_hub

from huggingface_hub import login, create_repo, upload_file, upload_folder

# Configuration
TOKEN = os.environ.get("HF_TOKEN") # Token removed for security
MODEL_REPO = "medicomrityunjay/DermaSentinel-Weights"
SPACE_REPO = "medicomrityunjay/DermaSentinel"

def deploy_huggingface_pro():
    print(f"🚀 Starting Deployment to Hugging Face...")
    
    # Login
    print("Logging in...")
    login(token=TOKEN)

    # --- Phase 1: The Vault (Model Repo) ---
    print(f"\n--- Phase 1: The Vault ({MODEL_REPO}) ---")
    # create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
    print("✅ Weights Secured in Model Hub (Skipped re-upload).")

    # --- Phase 2: The Refactor (Code Update) ---
    print(f"\n--- Phase 2: The Refactor (Skipped - Already Done) ---")
    
    # --- Phase 3: The Launch (Space Deployment) ---
    print(f"\n--- Phase 3: The Launch ({SPACE_REPO}) ---")
    create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="docker", exist_ok=True)
    
    print("Uploading Code Bundle (excluding local weights)...")
    upload_folder(
        folder_path=".",
        repo_id=SPACE_REPO,
        repo_type="space",
        ignore_patterns=[
            "core/models/**/weights/*.pth", 
            "__pycache__", 
            "*.git", 
            "deploy_huggingface_pro.py", 
            "finalize_project.py", 
            "audit_report.txt", 
            "logs/", 
            ".venv", 
            "venv",
            "*.pyc",
            "data/"
        ]
    )
    
    print(f"\nDeployment Complete. Your app is building at https://huggingface.co/spaces/{SPACE_REPO}")

if __name__ == "__main__":
    deploy_huggingface_pro()
