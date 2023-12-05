import argparse
import os
import torch
import json
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from PIL import Image
from going_modular.going_modular import engine
import torchvision
from torchinfo import summary
from helper_functions import set_seeds
import matplotlib.pyplot as plt

# Define an argument parser
parser = argparse.ArgumentParser(description="Train Vision Transformer on custom data")
parser.add_argument("--gpus", type=int, nargs="+", default=[0], help="List of GPU numbers to use (e.g., --gpus 0 1 2)")
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--save_dir", type=str, default="./saved_models", help="Directory to save model weights")
args = parser.parse_args()

# Use specified GPUs if available, otherwise use CPU
device = torch.device("cuda:" + ",".join(map(str, args.gpus)) if torch.cuda.is_available() and len(args.gpus) > 0 else "cpu")

# Define data directory
train_dir = 'C://Users//krish//Downloads//flowers_dataset//train'

# Get automatic transforms from pretrained ViT weights
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit_transforms = pretrained_vit_weights.transforms()

# Extract class names from subdirectories in train_dir
class_names = sorted(os.listdir(train_dir))

# Create a class_to_label dictionary to map class names to labels
# class_to_label = {class_name: i for i, class_name in enumerate(class_names)}
# label_to_class = {i: class_name for i, class_name in enumerate(class_names)}
# label_to_class = {"classes"}
# Save the label-to-class mapping as a JSON file
with open('label_to_class.json', 'w') as json_file:
    json.dump('label_to_class', json_file)
    # print("Class_dict in json")

# Create a DataLoader for training and validation
NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir, transform, batch_size, num_workers, split_ratios=(0.7, 0.15, 0.15)):
    full_dataset = datasets.ImageFolder(train_dir, transform=transform)

    # Calculate the sizes of the training, validation, and test sets
    num_samples = len(full_dataset)
    train_size = int(split_ratios[0] * num_samples)
    val_size = int(split_ratios[1] * num_samples)
    test_size = num_samples - train_size - val_size

    # Split the dataset into training, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size]
    )

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

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, test_dataloader

if __name__ == '__main__':
    # Setup data loaders
    train_dataloader_pretrained, val_dataloader_pretrained = create_dataloaders(train_dir=train_dir,
                                                                            transform=pretrained_vit_transforms,
                                                                            batch_size=32,
                                                                            num_workers=NUM_WORKERS)

    # 1. Get pretrained weights for ViT-Base


    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 


    # 2. Setup a ViT model instance with pretrained weights


    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)


    # Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    num_classes = len(class_names)

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     pretrained_vit = nn.DataParallel(pretrained_vit)

# Assuming you want to add 3 linear layers with hidden size 256
    hidden_size = 256
    num_additional_layers = 4
    input_channels = 768
    dropout_prob = 0.4

    # Get the output size of the last layer in the ViT model
    # vit_output_size = pretrained_vit.heads.in_features
    # Get the output size of the last layer in the ViT model
    vit_output_size = input_channels
    layers = []

    # Add the initial linear layer
    layers.append(nn.Linear(in_features=vit_output_size, out_features=hidden_size))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout_prob))

    # Add the specified number of additional layers
    for _ in range(num_additional_layers):
        layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))

    # Add the final output layer
    layers.append(nn.Linear(in_features=hidden_size, out_features=num_classes))

    # Create a sequential model with the specified layers
    additional_layers_model = nn.Sequential(*layers).to(device)
    pretrained_vit.heads = additional_layers_model

    # pretrained_vit.heads = nn.linear(in_features=768, out_features=num_classes).to(device)

    # Print model summary
    summary(model=pretrained_vit,
        input_size=(32, 3, 224, 224),  # (batch_size, color_channels, height, width)
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

    # Create optimizer and loss function
    optimizer = torch.optim.Adam(params=pretrained_vit.parameters(),
                                lr=0.5*1e-4)
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
        total_train_loss = 0
        train_dataloader_pretrained = tqdm(train_dataloader_pretrained, total=len(train_dataloader_pretrained))
        for batch in train_dataloader_pretrained:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            # print(labels)
            class_names_batch = [class_names[label] for label in labels]
            optimizer.zero_grad()
            outputs = pretrained_vit(inputs)
            # Compute loss and backpropagate
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total_train_samples += labels.size(0)
            correct_train_samples += (predicted == labels).sum().item()
            total_train_loss += loss.item()  

        train_accuracy = 100 * correct_train_samples / total_train_samples
        average_train_loss = total_train_loss / len(train_dataloader_pretrained)  # Calculate average training loss

        print(f"Training Accuracy: {train_accuracy:.2f}%")
        print(f"Average Training Loss: {average_train_loss:.4f}")

        # Validation phase
        pretrained_vit.eval()
        total_val_samples = 0
        correct_val_samples = 0
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader_pretrained:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = pretrained_vit(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val_samples += labels.size(0)
                correct_val_samples += (predicted == labels).sum().item()
                val_loss = loss_fn(outputs, labels)
                total_val_loss += val_loss.item()

        val_accuracy = 100 * correct_val_samples / total_val_samples
        average_val_loss = total_val_loss / len(val_dataloader_pretrained)  # Calculate average validation loss
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        print(f"Average Validation Loss: {average_val_loss:.4f}")


        # Save model weights after each epoch
        save_path = os.path.join(saved_models_dir, f"epoch_{epoch + 1}.pth")
        torch.save(pretrained_vit, save_path)
        print(f"Model weights saved at {save_path}")

    print("Training complete.")
