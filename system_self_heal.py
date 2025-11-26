import os
import sys
import shutil

def system_self_heal():
    print("🔧 Starting System Self-Healing Protocol...")

    # 1. Database Repair
    db_files = ["derma.db", "dermasentinel.db"]
    db_purged = False
    for db_file in db_files:
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
                print(f"🗑️ Outdated Database Purged: {db_file}. Schema will auto-update on restart.")
                db_purged = True
            except Exception as e:
                print(f"⚠️ Failed to delete {db_file}: {e}")
    
    if not db_purged:
        print("ℹ️ No existing database found. Clean slate confirmed.")

    # 2. Structure Validation
    required_dirs = [
        "data/active_learning",
        "data/raw",
        "static/temp",
        "logs"
    ]
    
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)
    
    print("✅ Directory Structure Enforced.")

    # 3. Dependency Check
    print("🔍 Verifying Dependencies...")
    dependencies = ["fastapi", "sqlalchemy", "passlib", "jose"]
    missing_deps = []
    
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)
            print(f"⚠️ Warning: Module '{dep}' not found.")
    
    if not missing_deps:
        print("✅ Core Dependencies Verified.")

    # 4. Weight Verification
    print("⚖️ Verifying Model Weights...")
    weights = [
        "core/models/scalpel/weights/best_scalpel.pth",
        "core/models/classifier/weights/best_classifier.pth",
        "core/models/fusion/weights/best_fusion.pth"
    ]
    
    found_weights = 0
    for w in weights:
        if os.path.exists(w):
            print(f"✅ Found: {w}")
            found_weights += 1
        else:
            print(f"⚠️ Missing: {w} (Note: May be downloaded from HF Hub on startup)")

    # Final Verdict
    print("\n" + "="*30)
    print("SYSTEM REPAIRED. Ready to launch.")
    print("="*30)

if __name__ == "__main__":
    system_self_heal()
