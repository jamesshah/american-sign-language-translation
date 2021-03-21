import joblib
import pandas as pd
import psutil
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchsummary import summary

from src.augments import Augments
from src.dataset import ImageDataset
from src.models import CustomModel
from src.trainer import Trainer

# Saved Label Encoder Model
lb = joblib.load('lb.pkl')

LEARNING_RATE = 0.001
VALID_SIZE = 0.2
NUM_CLASSES = len(lb.classes_)
TRAIN_PATH = 'dataset/train/'
BATCH_SIZE = 64
IMAGE_SIZE = (3, 128, 128)
DATA_FILE = 'image-data-file.csv'
EPOCHS = 5
DEVICE = "cuda:0" if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = psutil.cpu_count()


def run(epochs, device):

    # Load data file
    dfx = pd.read_csv(DATA_FILE)

    # Create train and valid splits
    df_train, df_valid = train_test_split(
        dfx, test_size=VALID_SIZE, random_state=42, stratify=dfx.target.values)

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_image_paths = TRAIN_PATH + df_train.image_path.values
    valid_image_paths = TRAIN_PATH + df_valid.image_path.values

    train_targets = df_train.target.values
    valid_targets = df_valid.target.values

    # Create train and valid datasets for creating dataloaders
    train_dataset = ImageDataset(
        image_paths=train_image_paths,
        targets=train_targets,
        augmentations=Augments.train_augments
    )

    valid_dataset = ImageDataset(
        image_paths=valid_image_paths,
        targets=valid_targets,
        augmentations=Augments.valid_augments
    )

    # Create train and valid dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # Load model and print summary of the model
    model = CustomModel(NUM_CLASSES).to(device)
    summary(model, input_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Define the Trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device
    )

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    # Train for provided epochs
    for epoch in range(epochs):
        print(f"{'-'*20} EPOCH: {epoch+1}/{epochs} {'-'*20}")

        # Train for one epoch
        train_epoch_loss, train_epoch_acc = trainer.train_one_epoch()

        # Validate for one epoch
        val_epoch_loss, val_epoch_acc = trainer.validate_one_epoch()

        # Append the loss and accuracy for history
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)

        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)

        print("Saving the model...\n")

        torch.save(model.state_dict(),
                   f'saved_models/model-train-acc-{int(train_acc[-1])}val-acc-{int(val_acc[-1])}.pth')

        print("Model Saved successfully.\n")


if __name__ == '__main__':
    run(EPOCHS, DEVICE)
