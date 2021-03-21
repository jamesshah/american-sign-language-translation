import warnings
import torch

from tqdm import tqdm

warnings.simplefilter("ignore")


class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, optimizer, loss_fn, device='cuda'):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_one_epoch(self):
        """
        Runs one epoch of training, backpropagation, optimization and gets train accuracy
        """
        print("Training...\n")
        self.model.train()

        running_loss = 0.0
        running_correct = 0

        tk = tqdm(self.train_dataloader,
                  total=int(len(self.train_dataloader)),
                  leave=True)

        for _, data in enumerate(tk):

            images, labels = data["image"].to(
                self.device), data["targets"].to(self.device)

            # Get predictions
            outputs = self.model(images)

            # Training
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)

            running_correct += (preds == labels).sum().item()

            tk_desc = f"Train Loss: {loss.item(): .4f}"
            tk.set_description(desc=tk_desc)

        train_loss = running_loss / len(self.train_dataloader.dataset)
        train_accuracy = (running_correct /
                          len(self.train_dataloader.dataset)) * 100.

        print(
            f"Train Acc: {train_accuracy:.4f}\n")

        return train_loss, train_accuracy

    def validate_one_epoch(self):
        """
        Runs one epoch of prediction and validation accuracy calculation
        """
        print("Validating...\n")
        self.model.eval()
        running_loss = 0.0
        running_correct = 0

        tk = tqdm(self.valid_dataloader,
                  total=int(len(self.valid_dataloader)),
                  leave=True)

        with torch.no_grad():
            for _, data in enumerate(tk):
                images, labels = data["image"].to(
                    self.device), data["targets"].to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                running_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)

                running_correct += (preds == labels).sum().item()
                tk_desc = f"Val Loss: {loss.item():.4f}"
                tk.set_description(desc=tk_desc)

            val_loss = running_loss / len(self.valid_dataloader.dataset)
            val_accuracy = 100. * \
                (running_correct / len(self.valid_dataloader.dataset))
            print(f'Val Acc: {val_accuracy:.2f}\n')

        return val_loss, val_accuracy
