import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

def list_datasets():
    api = KaggleApi()
    api.authenticate()
    print("Searching for 'isic 2018 task 1'...")
    datasets = api.dataset_list(search="isic 2018 task 1")
    for d in datasets:
        print(f"{d.ref}")

if __name__ == "__main__":
    list_datasets()
