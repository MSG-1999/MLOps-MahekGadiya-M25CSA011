from huggingface_hub import HfApi, login
import os

# ==============================
# change these values
# ==============================
hf_username = "MSG1999"        # e.g., m25csa011
repo_name = "ml-ops-3"      # e.g., distilbert-classifier
model_folder = "model/"                 # your local folder path
private = False                         # true if you want private repo
# ==============================


def main():
    api = HfApi()
    login("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")  
    repo_id = f"{hf_username}/{repo_name}"
    # create repo if not exists
    print("creating repo if it does not exist...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True
    )

    # upload entire folder
    print("uploading model folder to hugging face...")
    api.upload_folder(
        folder_path=model_folder,
        repo_id=repo_id,
        repo_type="model"
    )

    print("upload complete!")
    print(f"model available at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
