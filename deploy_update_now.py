import os
import sys
import subprocess

def install_deps():
    print("📦 Installing huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])

try:
    import huggingface_hub
except ImportError:
    install_deps()

from huggingface_hub import login, HfApi

TOKEN = os.environ.get("HF_TOKEN") # Token removed for security
REPO_ID = "medicomrityunjay/DermaSentinel"

def deploy():
    print("🚀 Starting Deployment Sequence...")
    
    # 1. Login
    print("🔑 Logging in to Hugging Face...")
    login(token=TOKEN)
    
    # 2. Initialize API
    api = HfApi()
    
    # 3. Upload Folder
    print(f"📤 Uploading code to {REPO_ID}...")
    api.upload_folder(
        folder_path=".",
        repo_id=REPO_ID,
        repo_type="space",
        ignore_patterns=[
            "core/models/**/weights/*.pth", 
            "__pycache__", 
            "*.git", 
            "deploy_huggingface_pro.py", 
            "deploy_update_now.py",
            "finalize_project.py", 
            "audit_report.txt", 
            "logs/", 
            ".venv", 
            "venv",
            "*.pyc",
            "data/",
            "derma.db",
            "system_self_heal.py",
            "verify_deployment.py",
            "rollback_auth.py",
            "fix_visibility.py",
            ".DS_Store"
        ],
        commit_message="Update: Remove Auth & Polish UI for Public Demo"
    )
    
    print("\n" + "="*50)
    print(f"🚀 Deployment Sent. The Space is rebuilding.")
    print(f"Monitor logs at: https://huggingface.co/spaces/{REPO_ID}")
    print("="*50)

if __name__ == "__main__":
    deploy()
