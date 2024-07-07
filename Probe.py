import torch, os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import ElasticNetCV ,MultiTaskElasticNetCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import r2_score, f1_score, accuracy_score

class DataHandler:
    def __init__(self, train_hidden_path, train_labels_path, 
                 test_hidden_path, test_labels_path, 
                 train_csv_path, test_csv_path, 
                 categorical_columns, numeric_columns,
                 is_numerical):
        # Load data
        try:
            self.train_features = torch.load(train_hidden_path).float().cpu().numpy()  # Convert to NumPy arrays here
            self.train_labels = torch.load(train_labels_path).float().cpu().numpy()
            self.test_features = torch.load(test_hidden_path).float().cpu().numpy()
            self.test_labels = torch.load(test_labels_path).float().cpu().numpy()
            self.is_numerical = is_numerical
        except RuntimeError as e:
            print("Error during tensor concatenation: " + str(e))
            print("ls path: " +str(os.listdir()))
            raise
        
        # Load and process CSV data
        if is_numerical:
            self.load_and_process_csv(train_csv_path, test_csv_path, numeric_columns)
        else:
            self.load_and_process_csv(train_csv_path, test_csv_path, categorical_columns)

        # Normalize features
        self.scaler = StandardScaler()
        self.train_features = self.scaler.fit_transform(self.train_features)
        self.test_features = self.scaler.transform(self.test_features)

    def get_dataloaders(self, batch_size=32):
        # Create TensorDatasets
        train_dataset = TensorDataset(torch.from_numpy(self.train_features).float(), torch.from_numpy(self.train_labels).float())
        test_dataset = TensorDataset(torch.from_numpy(self.test_features).float(), torch.from_numpy(self.test_labels).float())

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def get_dataloaders_for_each_label(self, batch_size=32, attributes=None):
        """
        This function creates a separate DataLoader for each label attribute.
        :param batch_size: Size of each batch.
        :param attributes: List of attribute names corresponding to each column in labels.
        :return: Dictionary of DataLoaders for each attribute for both training and testing.
        """
        train_loaders = {}
        test_loaders = {}

        for i, attr in enumerate(attributes):
            # Create datasets for each attribute
            train_dataset = TensorDataset(torch.from_numpy(self.train_features).float(), torch.from_numpy(self.train_labels[:, i]).float())
            test_dataset = TensorDataset(torch.from_numpy(self.test_features).float(), torch.from_numpy(self.test_labels[:, i]).float())
            
            # Create loaders for each attribute
            train_loaders[attr] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loaders[attr] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loaders, test_loaders

    def load_and_process_csv(self, train_csv_path, test_csv_path, columns):
        # Load CSV files
        train_csv_data = pd.read_csv(train_csv_path, usecols=columns)
        test_csv_data = pd.read_csv(test_csv_path, usecols=columns)

        # Convert categorical data to numerical if necessary
        train_csv_data = self.encode_features(train_csv_data, columns)
        test_csv_data = self.encode_features(test_csv_data, columns )

        if self.is_numerical:
            # Convert to NumPy and concatenate with existing features
            self.train_labels = np.hstack((self.train_labels, train_csv_data))
            self.test_labels = np.hstack((self.test_labels, test_csv_data))
        else:
            self.train_labels =  train_csv_data
            self.test_labels =  test_csv_data

    
    def encode_features(self, df, columns):
        # Convert categorical data to numeric codes and ensure all data is appropriate for tensor conversion
        for col in columns:
            if self.is_numerical:
                df[col] = df[col].astype(np.float32)  # Ensure numeric columns are also in float32
            else:
                df[col] = pd.Categorical(df[col]).codes 
        return df.values  # Return as numpy array
    
    def print_first_rows(self, num_rows=5):
        print("First rows of train labels:")
        print(self.train_labels[:num_rows])
        print("First rows of test labels:")
        print(self.test_labels[:num_rows])


class MultiElasticNetModel:
    def __init__(self):
        self.model = None

    def train(self, train_features, train_labels):
        self.model = MultiTaskElasticNetCV(cv=5, random_state=0, max_iter=10000, alphas=[0.1, 1, 10], l1_ratio=[0.2, 0.5, 0.8])
        self.model.fit(train_features, train_labels)

    def evaluate(self, test_features, test_labels):
        predictions = self.model.predict(test_features)
        return r2_score(test_labels, predictions)


class ElasticNetModel:
    def __init__(self, attributes):
        # Maintain a list of models if you have multiple targets
        self.models = []
        self.attributes = attributes
        self.train_predictions = []

    def train(self, train_features, train_labels):
        # Train a separate model for each column in train_labels
        for i in range(train_labels.shape[1]):
            model = ElasticNetCV(cv=5, random_state=0, max_iter=10000, alphas=[0.1, 1, 10], l1_ratio=[0.2, 0.5, 0.8])
            model.fit(train_features, train_labels[:, i])
            self.models.append(model)

            # Store training predictions
            self.train_predictions.append(model.predict(train_features))

    def evaluate(self, features, labels, training=False):
        # Evaluate each model and return the R2 score for each
        predictions_list = self.train_predictions if training else []
        r2_scores = {}
        for i, model in enumerate(self.models):
            if not training:
                predictions = model.predict(features)
                predictions_list.append(predictions)
            else:
                predictions = predictions_list[i]
            r2 = r2_score(labels[:, i], predictions)
            r2_scores[self.attributes[i]] = r2
        return r2_scores


class LogisticModel:
    def __init__(self, attributes):
        self.models = []
        self.attributes = attributes
        self.train_predictions = []

    def train(self, train_features, train_labels):
        """ Train a separate logistic regression model for each categorical label. """
        for i, attr in enumerate(self.attributes):
            # Using a logistic regression model with cross-validation
            # model = LogisticRegressionCV(cv=5, random_state=0, max_iter=10000, multi_class='multinomial', solver='lbfgs')
            model = LogisticRegressionCV(cv=5, random_state=0, max_iter=10000, multi_class='multinomial', solver='saga', penalty='l2', Cs=[0.1, 1, 10])
           
            model.fit(train_features, train_labels[:, i])
            self.models.append(model)

            # Store training predictions for later evaluation
            self.train_predictions.append(model.predict(train_features))

    def evaluate(self, features, labels, training=False):
        """ Evaluate each model and return accuracy and F1 score for each. """
        predictions_list = self.train_predictions if training else []
        accuracy_scores = {}
        f1_scores = {}
        for i, model in enumerate(self.models):
            if not training:
                predictions = model.predict(features)
                predictions_list.append(predictions)
            else:
                predictions = predictions_list[i]

            accuracy = accuracy_score(labels[:, i], predictions)
            f1 = f1_score(labels[:, i], predictions, average='weighted')
            accuracy_scores[self.attributes[i]] = accuracy
            f1_scores[self.attributes[i]] = f1

        return accuracy_scores, f1_scores


class MultiMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiMLP, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


class MultiMLPModel:
    def __init__(self, input_size, output_size):
        self.model = MultiMLP(input_size, output_size)
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


class SimpleMLP(nn.Module):
    def __init__(self, input_size):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Linear(input_size, 1)  # Output size is 1 for single target regression

    def forward(self, x):
        return self.fc(x)


class MLPModel:
    def __init__(self, input_size, attributes, learning_rate=0.001):
        self.models = {attr: SimpleMLP(input_size) for attr in attributes}  # Create a model for each attribute
        self.criterion = nn.MSELoss()
        self.optimizers = {attr: optim.Adam(self.models[attr].parameters(), lr=learning_rate) for attr in attributes}
        self.attributes = attributes

    def train(self, train_loaders, device='cpu', epochs=50):
        for attr in self.attributes:
            model = self.models[attr].to(device)
            optimizer = self.optimizers[attr]
            for epoch in range(epochs):
                model.train()
                for inputs, targets in train_loaders[attr]:  # Assume train_loaders is a dict with loaders for each attribute
                    inputs, targets = inputs.to(device), targets[:, None].to(device)  # Ensure targets are reshaped properly
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                print(f'Epoch {epoch + 1}, Loss for {attr}: {loss.item()}')

    def evaluate(self, test_loaders, device='cpu'):
        predictions = {}
        r2_scores = {}
        for attr in self.attributes:
            model = self.models[attr].eval().to(device)
            all_outputs = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in test_loaders[attr]:
                    inputs, labels = inputs.to(device), labels[:, None].to(device)
                    outputs = model(inputs)
                    all_outputs.append(outputs.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
            all_outputs = np.concatenate(all_outputs, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            r2_scores[attr] = r2_score(all_labels, all_outputs)
            predictions[attr] = all_outputs
        return predictions, r2_scores


def main():
    attributes = ['predict_event', 'pleasantness', 'attention', 'other_responsibility', 'chance_control', 'social_norms']

    categorical_columns = ['gender', 'education', 'ethnicity', 'event_duration', 'emotion_duration']
    categorical_columns = ['education']
    numeric_columns = ['intensity', 'age', 'extravert', 'critical', 
                'dependable', 'anxious', 'open', 'quiet', 'sympathetic', 'disorganized', 'calm', 'conventional']

    is_numerical = False
    # categorical_columns = ['round_number']
    # numeric_columns = ['intensity']

    data_handler = DataHandler('output/train_hidden_states.pt', 'output/train_appraisal_labels.pt', 
                               'output/test_hidden_states.pt', 'output/test_appraisal_labels.pt',
                               'data/enVent_new_Data_train.csv', 'data/enVent_new_Data_test.csv', 
                               categorical_columns, numeric_columns, is_numerical)
    data_handler.print_first_rows()
    train_loader, test_loader = data_handler.get_dataloaders()


    if is_numerical:
        separate_train_loaders, separate_test_loaders = data_handler.get_dataloaders_for_each_label(batch_size=32, attributes=attributes+numeric_columns)
        # # Multi ElasticNet Model
        # en_model_multi = MultiElasticNetModel()
        # en_model_multi.train(data_handler.train_features, data_handler.train_labels)
        # r2_elastic = en_model_multi.evaluate(data_handler.test_features, data_handler.test_labels)
        # print(f"ElasticNet R2 Score: {r2_elastic}")

        # ElasticNet Model
        en_model = ElasticNetModel(attributes+numeric_columns)
        en_model.train(data_handler.train_features, data_handler.train_labels)
        print("Training done:\n")

        train_r2_scores = en_model.evaluate(data_handler.train_features, data_handler.train_labels, training=True)
        test_r2_scores = en_model.evaluate(data_handler.test_features, data_handler.test_labels, training=False)

        # Printing R2 scores
        print("Training R² Scores:")
        for attribute, r2_score in train_r2_scores.items():
            print(f"{attribute}: {r2_score:.3f}")

        print("\nTesting R² Scores:")
        for attribute, r2_score in test_r2_scores.items():
            print(f"{attribute}: {r2_score:.3f}")

    else:
        separate_train_loaders, separate_test_loaders = data_handler.get_dataloaders_for_each_label(batch_size=32, attributes=categorical_columns)
        categorical_model = LogisticModel(categorical_columns)
        categorical_model.train(data_handler.train_features, data_handler.train_labels)
        print("Training done:\n")
        train_accuracy, train_f1 = categorical_model.evaluate(data_handler.train_features, data_handler.train_labels, training=True)
        test_accuracy, test_f1 = categorical_model.evaluate(data_handler.test_features, data_handler.test_labels, training=False)

        # Printing accuracy and F1 scores
        print("Training Accuracy and F1 Scores:")
        for attribute in categorical_columns:
            print(f"{attribute} - Accuracy: {train_accuracy[attribute]:.3f}, F1 Score: {train_f1[attribute]:.3f}")

        print("\nTesting Accuracy and F1 Scores:")
        for attribute in categorical_columns:
            print(f"{attribute} - Accuracy: {test_accuracy[attribute]:.3f}, F1 Score: {test_f1[attribute]:.3f}")





    # # Multi MLP Model
    # mlp_model = MultiMLPModel(input_size=data_handler.train_features.shape[1], output_size=data_handler.train_labels.shape[1])
    # mlp_model.train(train_loader)
    # mlp_predictions = mlp_model.evaluate(test_loader)
    # r2_mlp = r2_score(data_handler.test_labels, mlp_predictions)
    # print(f"MLP R2 Score: {r2_mlp}")

    # model = MLPModel(input_size=data_handler.train_features.shape[1], attributes=attributes, learning_rate=0.01)
    # model.train(separate_train_loaders, device=device, epochs=20)
    # predictions, r2_scores = model.evaluate(separate_test_loaders, device=device)
    # for attr, score in r2_scores.items():
    #     print(f'R2 score for {attr}: {score:.4f}')

            
if __name__ == "__main__":
    main()

