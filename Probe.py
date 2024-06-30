import torch
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch.nn as nn
import torch.optim as optim

def train_elasticnet(hidden_states, labels, alpha=0.1, l1_ratio=0.5):
    """ Train an ElasticNet model on the provided hidden states and labels. """
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(hidden_states, labels)
    return model

class SimpleMLP(nn.Module):
    """ A simple one-layer MLP for regression tasks. """
    def __init__(self, input_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

def train_mlp(hidden_states, labels, epochs=50, learning_rate=0.01):
    """ Train a simple MLP on the provided hidden states and labels. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP(hidden_states.shape[1], labels.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    hidden_states, labels = torch.tensor(hidden_states, dtype=torch.float32).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(hidden_states)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return model

def load_and_train_models(hidden_file_path, labels_file_path):
    # Load hidden states and labels
    hidden_states = torch.load(hidden_file_path)
    labels = torch.load(labels_file_path)

    # Convert tensors to numpy for sklearn compatibility
    hidden_states_np = hidden_states.numpy()
    labels_np = labels.numpy()

    # Train ElasticNet
    elastic_model = train_elasticnet(hidden_states_np, labels_np)
    print("Trained ElasticNet Model")

    # Train MLP
    mlp_model = train_mlp(hidden_states, labels)
    print("Trained MLP Model")

    # Evaluate models
    predicted_labels_elastic = elastic_model.predict(hidden_states_np)
    predicted_labels_mlp = mlp_model(torch.tensor(hidden_states_np, dtype=torch.float32)).detach().numpy()

    r2_elastic = r2_score(labels_np, predicted_labels_elastic)
    r2_mlp = r2_score(labels_np, predicted_labels_mlp)

    print(f"ElasticNet R2 Score: {r2_elastic}")
    print(f"MLP R2 Score: {r2_mlp}")

    return elastic_model, mlp_model, r2_elastic, r2_mlp

# Example usage:
# elastic_model, mlp_model, r2_elastic, r2_mlp = load_and_train_models('hidden_states.pt', 'appraisal_labels.pt')
