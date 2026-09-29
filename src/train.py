import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset  # <-- Added TensorDataset
from torchvision import models
import mlflow
import mlflow.pytorch

def train_model():
    # 1. Load Parameters
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    lr = config["train"]["learning_rate"]
    batch_size = config["train"]["batch_size"]
    epochs = config["train"]["epochs"]
    data_dir = config["data"]["data_dir"]

    # 2. Setup Data Loaders (Reading our tiny real image subset)
    dataset_path = os.path.join(data_dir, "real_dataset.pt")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Could not find {dataset_path}. Please run 'dvc repro' or execute src/prepare_data.py first.")
        
    data_dict = torch.load(dataset_path)
    
    # Wrap the pre-loaded real tensors into a dataset
    train_set = TensorDataset(data_dict["images"], data_dict["labels"])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 3. Define Simple CNN Model
    model = models.resnet18(num_classes=10) #  ip, n_cnn (frozen), fully connected layer(trainable parameters)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. MLflow Experiment Tracking
    mlflow.set_experiment("CIFAR10_Deep_Learning")
    
    with mlflow.start_run():
        # Log parameters automatically from params.yaml
        mlflow.log_params(config["train"])
        mlflow.log_params(config["data"])
        
        # Capture current DVC data state (Lineage tracking)
        if os.path.exists("dvc.lock"):
            with open("dvc.lock", "r") as lock_f:
                mlflow.set_tag("dvc_lock", lock_f.read())

        print("Starting training loop...")
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs) # resnet - extract features but no learnable parameters
                loss = criterion(outputs, labels) # loss criterion
                loss.backward() 
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = correct / total
            
            # Log metrics per epoch to MLflow
            mlflow.log_metric("loss", float(epoch_loss), step=int(epoch + 1))
            mlflow.log_metric("accuracy", float(epoch_acc), step=int(epoch + 1))
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

        # Log the trained model weights directly to MLflow
        mlflow.pytorch.log_model(model, artifact_path="model",serialization_format="pickle")
        print("Model tracking complete.")

if __name__ == "__main__":
    train_model()
