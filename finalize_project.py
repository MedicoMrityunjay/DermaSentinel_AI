import os
import shutil

def finalize_project():
    root_dir = os.getcwd()
    print(f"Cleaning up project in: {root_dir}")
    
    # 1. Delete Temporary Scripts
    prefixes = ('init_', 'upgrade_', 'audit_', 'verify_', 'execute_', 'rescue_')
    
    files_to_delete = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f.startswith(prefixes)]
    
    print(f"\nDeleting {len(files_to_delete)} temporary scripts...")
    for f in files_to_delete:
        try:
            os.remove(os.path.join(root_dir, f))
            print(f"Deleted: {f}")
        except OSError as e:
            print(f"Error deleting {f}: {e}")

    # 2. Deep Clean Caches
    print("\nDeep cleaning caches...")
    for root, dirs, files in os.walk(root_dir):
        # Delete directories
        for d in list(dirs):
            if d in ('__pycache__', '.ipynb_checkpoints'):
                path = os.path.join(root, d)
                try:
                    shutil.rmtree(path)
                    print(f"Deleted folder: {path}")
                    dirs.remove(d) # Don't traverse into deleted dir
                except OSError as e:
                    print(f"Error deleting folder {path}: {e}")
        
        # Delete files
        for f in files:
            if f == '.DS_Store':
                path = os.path.join(root, f)
                try:
                    os.remove(path)
                    print(f"Deleted file: {path}")
                except OSError as e:
                    print(f"Error deleting file {path}: {e}")

    # 3. Reset Session Storage
    print("\nResetting session storage...")
    
    def clean_folder_contents(folder_path):
        if os.path.exists(folder_path):
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                        print(f"Deleted: {item_path}")
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        print(f"Deleted folder: {item_path}")
                except Exception as e:
                    print(f"Failed to delete {item_path}. Reason: {e}")
        else:
            print(f"Folder not found (skipping): {folder_path}")

    clean_folder_contents(os.path.join(root_dir, 'static', 'temp'))
    clean_folder_contents(os.path.join(root_dir, 'data', 'active_learning'))

    # 4. Verification Report
    print("\nRemaining files in root:")
    remaining_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
    for f in remaining_files:
        print(f" - {f}")

    print("\nProject Cleaned & Ready for Distribution.")

if __name__ == "__main__":
    finalize_project()
