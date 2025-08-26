import os
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset

class ModelDataset(Dataset):
    def __init__(self, data):

        self.data = data

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        x, y = self.data[idx]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class BuildModel:
    def __init__(self, model, data, identifier):
        self.model = model
        self.train_data, self.val_data, self.test_data = self.data_splits(data)
        self.identifier = identifier
        self.loss, self.r2 = -1, -1

    def data_splits(self, data : pd.core.frame.DataFrame) -> list:
    
        target_idx = random.randint(0, data.shape[1])

        setattr(self, 'target', data.columns[target_idx])
        setattr(self, 'features', [np.delete(data.columns.to_numpy(), target_idx, axis = 0)])

        data = data.to_numpy()

        targets = data[:, target_idx]
        features = np.delete(data, target_idx, axis = 1)
        dataset = [[feature, target] for feature, target in zip(features, targets)]

        random.shuffle(dataset)

        train_ratio = 0.6
        split_index = int(len(dataset) * train_ratio)

        train_data = dataset[:split_index]
        test_val_data = dataset[split_index:]

        random.shuffle(test_val_data)
        val_split_index = int(len(test_val_data) * 0.5)

        test_data = test_val_data[:val_split_index]
        val_data  = test_val_data[val_split_index:]

        return train_data, val_data, test_data

    def training_loop(self, num_epochs = 10, batch_size = 1024, learning_rate = 1e-2, device = "cpu") -> None:

        dataset = ModelDataset(self.train_data)
        dataloader = DataLoader(dataset, batch_size, shuffle=True)

        self.model = self.model.to(device)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0

            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = loss_function(output, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            print(f"Epoch {epoch+1}/{num_epochs} | MSE: {avg_loss:.4f}")
            self.no_grad_loop("Validation", self.val_data, batch_size, device)

    def no_grad_loop(self, type_ : str, data : np.ndarray, batch_size = 1024, device = "cpu") -> list:

        dataset = ModelDataset(data)
        dataloader = DataLoader(dataset, batch_size, shuffle=False)

        loss_function = nn.MSELoss()
        total_loss = 0.0
        all_preds, all_targets = [], []

        self.model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)

                output = self.model(x)
                loss = loss_function(output, y)
                total_loss += loss.item()

                all_preds.append(output.cpu())
                all_targets.append(y.cpu())

        avg_loss = total_loss / len(dataloader)

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        r2 = r2_score(all_targets, all_preds)

        print(f"{type_} MSE: {avg_loss:.4f} | {type_} R²: {r2:.4f}")

        self.loss = avg_loss
        self.r2 = r2
    
    def report(self) -> None:

        model_path = os.path.join("models", f"{self.identifier}_model.pth")

        torch.save(self.model.state_dict(), model_path)
    
        outputs_path = os.path.join("outputs", f"{self.identifier}_outputs.txt")

        with open(outputs_path, "w") as file:
            file.write(
                f"MLP Model {self.identifier} \n Features: {self.features} | Target: {self.target} \n Testing Performance | MSE = {self.loss} | R2 Score = {self.loss}")