import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score

class DataHandler:
    def __init__(self, train_hidden_path, train_labels_path, test_hidden_path, test_labels_path):
        # Load data
        self.train_features = torch.load(train_hidden_path).float()
        self.train_labels = torch.load(train_labels_path).float()
        self.test_features = torch.load(test_hidden_path).float()
        self.test_labels = torch.load(test_labels_path).float()

        # Normalize features
        self.scaler = StandardScaler()
        self.train_features = self.scaler.fit_transform(self.train_features.numpy())
        self.test_features = self.scaler.transform(self.test_features.numpy())

    def get_dataloaders(self, batch_size=32):
        # Create TensorDatasets
        train_dataset = TensorDataset(torch.from_numpy(self.train_features).float(), torch.from_numpy(self.train_labels).float())
        test_dataset = TensorDataset(torch.from_numpy(self.test_features).float(), torch.from_numpy(self.test_labels).float())

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    

class SimpleMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


class MLPModel:
    def __init__(self, input_size, output_size):
        self.model = SimpleMLP(input_size, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = None

    def train(self, train_loader, device='cpu', epochs=50):
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        for epoch in range(epochs):
            self.model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

    def evaluate(self, test_loader, device='cpu'):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                predictions.append(outputs.cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        return predictions


class ElasticNetModel:
    def __init__(self):
        self.model = None

    def train(self, train_features, train_labels):
        self.model = ElasticNetCV(cv=5, random_state=0)
        self.model.fit(train_features, train_labels)

    def evaluate(self, test_features, test_labels):
        predictions = self.model.predict(test_features)
        return r2_score(test_labels, predictions)
    
    
def main():
    data_handler = DataHandler('train_hidden_states.pt', 'train_appraisal_labels.pt', 'test_hidden_states.pt', 'test_appraisal_labels.pt')
    train_loader, test_loader = data_handler.get_dataloaders()

    # ElasticNet Model
    en_model = ElasticNetModel()
    en_model.train(data_handler.train_features, data_handler.train_labels)
    r2_elastic = en_model.evaluate(data_handler.test_features, data_handler.test_labels)
    print(f"ElasticNet R2 Score: {r2_elastic}")

    # MLP Model
    mlp_model = MLPModel(input_size=data_handler.train_features.shape[1], output_size=data_handler.train_labels.shape[1])
    mlp_model.train(train_loader)
    mlp_predictions = mlp_model.evaluate(test_loader)
    r2_mlp = r2_score(data_handler.test_labels, mlp_predictions)
    print(f"MLP R2 Score: {r2_mlp}")

if __name__ == "__main__":
    main()

