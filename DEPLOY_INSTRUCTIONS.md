# Deploying DermaSentinel to Hugging Face Spaces

## Prerequisites
- A Hugging Face account.
- Git installed locally.
- Git LFS (Large File Storage) installed: `git lfs install`

## Steps

1.  **Create a Space:**
    - Go to [Hugging Face Spaces](https://huggingface.co/spaces).
    - Click "Create new Space".
    - Enter a name (e.g., `DermaSentinel`).
    - Select **Docker** as the SDK.
    - Choose "Public" or "Private".
    - Click "Create Space".

2.  **Clone the Repository:**
    - On your local machine, clone the newly created space:
      ```bash
      git clone https://huggingface.co/spaces/<YOUR_USERNAME>/DermaSentinel
      cd DermaSentinel
      ```

3.  **Prepare Files:**
    - Copy all files from your local `DermaSentinel` project directory into this cloned repository folder.
    - **Crucial:** Ensure your model weights (`.pth` files) are included.

4.  **Track Large Files:**
    - Initialize Git LFS and track the model weights:
      ```bash
      git lfs install
      git lfs track "*.pth"
      git lfs track "*.safetensors"
      ```

5.  **Push to Deploy:**
    - Add, commit, and push your changes:
      ```bash
      git add .
      git commit -m "Initial deployment of DermaSentinel"
      git push
      ```

6.  **Monitor Build:**
    - Go back to your Space page on Hugging Face.
    - Click on the "Logs" tab to watch the build process.
    - Once the status changes to "Running", your app is live!

## Troubleshooting
- **Permission Errors:** The Dockerfile creates a non-root user (ID 1000) to comply with HF security policies. Ensure your code writes temporary files to directories where this user has permissions (e.g., `/tmp` or the working directory).
- **OOM (Out of Memory):** If the build fails or the app crashes, you might need to upgrade the Space hardware (Settings -> Hardware) to a GPU instance, especially since we are running deep learning models.
