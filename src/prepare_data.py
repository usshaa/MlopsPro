import os
import yaml
import torch
import shutil
from torchvision import datasets

def prepare_data():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    data_dir = config["data"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    
    # Move the download directory OUTSIDE your project workspace root
    # This creates a hidden global cache folder on your machine ('~/.cifar10_cache')
    cache_dir = os.path.expanduser("~/.cifar10_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    print("Checking local cache for real CIFAR-10 images...")
    # PyTorch will check 'cache_dir'. If it exists, it skips downloading instantly!
    full_dataset = datasets.CIFAR10(root=cache_dir, train=True, download=True) # 170 MB
    
    # Extract only the first 200 real images and labels
    real_images = torch.tensor(full_dataset.data[:300]).permute(0, 3, 1, 2).float() / 255.0
    real_labels = torch.tensor(full_dataset.targets[:300])
    
    # Save the tiny sub-slice (<1MB) into the clean DVC directory
    torch.save({"images": real_images, "labels": real_labels}, os.path.join(data_dir, "real_dataset.pt"))
    print("Real subset extracted! Data size reduced to <1MB.")
    print("Raw files preserved in local machine cache directory for instant future re-runs.")

if __name__ == "__main__":
    prepare_data()