import argparse
import os
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from PIL import Image
import torchvision
from torchinfo import summary
import matplotlib.pyplot as plt

# Define an argument parser
parser = argparse.ArgumentParser(description="Train Vision Transformer on custom data")
# parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1], help="List of GPU numbers to use (e.g., --gpus 0 1 2)")
parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
parser.add_argument("--save_dir", type=str, default="./saved_models", help="Directory to save model weights")
args = parser.parse_args()

# Use specified GPUs if available, otherwise use CPU
# device = torch.device("cuda:" + ",".join(map(str, args.gpus)) if torch.cuda.is_available() and len(args.gpus) > 0 else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() and len([0,1]) > 0 else "cpu")
print(device)


# Define data directory
train_dir = "C://Users//krish//Downloads//flowers_dataset//train"

# Get automatic transforms from pretrained ViT weights
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit_transforms = pretrained_vit_weights.transforms()

# Extract class names from subdirectories in train_dir
class_names = sorted(os.listdir(train_dir))

# Create a DataLoader for training and validation
NUM_WORKERS = os.cpu_count()
print(NUM_WORKERS)

def create_dataloaders(train_dir, transform, batch_size, num_workers, validation_split=0.2):
    full_dataset = datasets.ImageFolder(train_dir, transform=transform)

    # Calculate the size of the validation set
    num_samples = len(full_dataset)
    val_size = int(validation_split * num_samples)
    train_size = num_samples - val_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Turn datasets into data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_dataloader, val_dataloader

if __name__ == '__main__':
    # Setup data loaders
    train_dataloader_pretrained, val_dataloader_pretrained = create_dataloaders(train_dir=train_dir,
                                                                            transform=pretrained_vit_transforms,
                                                                            batch_size=2,  # Adjust batch size for multiple GPUs
                                                                            num_workers=NUM_WORKERS)

    # 1. Get pretrained weights for ViT-Base
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

    # 2. Setup a ViT model instance with pretrained weights
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

    if len([0]) > 1:  # Use nn.DataParallel for multiple GPUs
        print(f"Using {len([0,1])} GPUs")
        pretrained_vit = nn.DataParallel(pretrained_vit, device_ids=[0])
    # pretrained_vit.module.head.requires_grad = True
    pretrained_vit = pretrained_vit.to(device)
    # 3. Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    # Customize the classifier head based on the number of classes
    num_classes = len(class_names)
    print(num_classes)
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=num_classes).to(device)
    # last_layer = pretrained_vit..heads
    # last_layer.requires_grad = True  # Ensure the last layer is trainable
    for parameter in pretrained_vit.heads.parameters():
        parameter.requires_grad =True

    # Print model summary
    summary(model=pretrained_vit,
            input_size=(32, 3, 224, 224),  # (batch_size, color_channels, height, width)
            col_names=["input_size", "output_size", "num_params", "trainable"]
    )

    # Create optimizer and loss function
    optimizer = torch.optim.Adam(params=pretrained_vit.parameters(),
                                 lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    saved_models_dir = args.save_dir  # Directory to save model weights
    os.makedirs(saved_models_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Training phase
        pretrained_vit.train()
        total_train_samples = 0
        correct_train_samples = 0
        train_dataloader_pretrained = tqdm(train_dataloader_pretrained, total=len(train_dataloader_pretrained))
        for batch in train_dataloader_pretrained:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            optimizer.zero_grad()
            outputs = pretrained_vit(inputs)
            # Compute loss and backpropagate
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total_train_samples += labels.size(0)
            correct_train_samples += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train_samples / total_train_samples
        print(f"Training Accuracy: {train_accuracy:.2f}%")
        print(f"Training Loss: {loss:.2f}%")

        # Validation phase
        pretrained_vit.eval()
        total_val_samples = 0
        correct_val_samples = 0
        with torch.no_grad():
            for batch in val_dataloader_pretrained:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = pretrained_vit(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val_samples += labels.size(0)
                correct_val_samples += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val_samples / total_val_samples
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        print(f"Validation Loss: {loss:.2f}%")

        # Save model weights after each epoch
        save_path = os.path.join(saved_models_dir, f"epoch_{epoch + 1}.pth")
        torch.save(pretrained_vit.state_dict(), save_path)  # Save model state_dict
        print(f"Model weights saved at {save_path}")

    print("Training complete.")