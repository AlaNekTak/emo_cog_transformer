import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import logging
import os
from datetime import datetime
from tqdm import tqdm

# Define the dataset class for handling text data
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

# Log class to handle logging activities
class Log:
    def __init__(self):
        filename = f'probe_date-{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}.txt'
        self.log_path = os.path.join('', filename)
        self.logger = self._setup_logging()

    def _setup_logging(self):
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            handlers=[
                                logging.FileHandler(self.log_path),
                                logging.StreamHandler()
                            ])
        return logging.getLogger()

# Function to extract hidden states from the model
# def extract_hidden_states(batch_texts, tokenizer, model, max_length=512):

#     inputs = tokenizer(
#         batch_texts,
#         padding='max_length',
#         truncation=True,
#         max_length=max_length,
#         return_tensors="pt"
#     ).to(model.device)

#     with torch.no_grad():
#         outputs = model(**inputs, output_hidden_states=True)

#     # Ensure outputs.hidden_states[-1] is a tensor before calling detach()
#     if isinstance(outputs.hidden_states[-1], torch.Tensor):
#         # Extract the last token of the last hidden state for each sequence
#         last_token_states = outputs.hidden_states[-1][:, -1, :].detach().cpu().numpy()
#     else:
#         logger.error("Expected outputs.hidden_states[-1] to be a PyTorch tensor.")
#         raise TypeError("outputs.hidden_states[-1] is not a PyTorch tensor.")

#     return last_token_states


def extract_hidden_states(batch_texts, tokenizer, model, max_length=512, logger=None, debug=False):
    """
    Extracts the hidden states of the last non-padded token for each input text batch.

    Args:
    - batch_texts: List of text strings to process.
    - tokenizer: Tokenizer object from the transformers library.
    - model: Pretrained model from the transformers library.
    - max_length: Maximum length of the tokenized input (default is 512).
    - logger: Logger object for logging information.
    - debug: If True, prints detailed information for debugging purposes.

    Returns:
    - A numpy array of the hidden states of the last non-padded token from the model for each input text.
    """
    # Tokenize the batch of texts
    inputs = tokenizer(
        batch_texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(model.device)

    # Run the model to get hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get the last hidden state from the model outputs
    hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, seq_length, hidden_size]

    # Get the attention mask to identify non-padded tokens
    attention_mask = inputs['attention_mask']  # Shape: [batch_size, seq_length]

    # Compute the sequence lengths (number of non-padded tokens for each sequence)
    sequence_lengths = attention_mask.sum(dim=1)  # Shape: [batch_size]
    # Indices of the last non-padded tokens
    last_token_indices = sequence_lengths - 1  # Subtract 1 for zero-based indexing

    # Initialize a list to collect the hidden states of the last non-padded tokens
    last_token_states = []

    # Extract the hidden state for the last non-padded token in each sequence
    for i in range(hidden_states.size(0)):
        index = last_token_indices[i].item()
        last_hidden_state = hidden_states[i, index, :]  # Shape: [hidden_size]
        last_token_states.append(last_hidden_state)

        if debug and i < 2:  # Print debug information for the first two examples
            input_ids = inputs['input_ids'][i]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            logger.info(f"Example {i+1}:")
            logger.info(f"Input text: {batch_texts[i]}")
            logger.info(f"Tokens: {tokens}")
            logger.info(f"Attention mask: {attention_mask[i].cpu().numpy()}")
            logger.info(f"Sequence length (excluding padding): {sequence_lengths[i].item()}")
            logger.info(f"Last token index: {index}")
            logger.info(f"Last token: {tokens[index]}")
            logger.info("-" * 50)

    # Stack the list into a tensor and convert to numpy array
    last_token_states = torch.stack(last_token_states).detach().cpu().numpy()  # Shape: [batch_size, hidden_size]

    return last_token_states


# Process batches of text to get hidden states
def process_batches(dataloader, tokenizer, model, logger):
    total_batches = len(dataloader)
    all_hidden_states = []
    for i, batch_texts in enumerate(tqdm(dataloader, desc="Processing batches"), 1):
        if i % 20 == 0 or i == total_batches:
            logger.info(f"Completed {i}/{total_batches} batches")
        hidden_states = extract_hidden_states(batch_texts, tokenizer, model)
        # hidden_states is already a NumPy array
        all_hidden_states.append(hidden_states)  # Keep as NumPy arrays for now

    # Concatenate all arrays in the list to a single NumPy array
    all_hidden_states_array = np.concatenate(all_hidden_states, axis=0)

    # Log the size of the hidden states array
    current_size = all_hidden_states_array.nbytes / (1024 ** 2)  # size in megabytes
    logger.info(f"Total size of hidden states array: {current_size:.2f} MB")

    # Save the array to a file
    np.save('hidden_states.npy', all_hidden_states_array)
    logger.info("Saved all hidden states as a .npy file")

    return all_hidden_states_array

# Probing function to analyze hidden states using regression
def probe(attributes, train_data, all_hidden_states, logger):
    """
    Probes each attribute separately using regression on the hidden states.

    Args:
    - attributes: List of attribute names to probe.
    - train_data: The DataFrame containing the target attributes.
    - all_hidden_states: NumPy array or PyTorch tensor of hidden states.
    - logger: Logger object for logging information.
    """
    # Ensure all_hidden_states is a NumPy array
    if isinstance(all_hidden_states, torch.Tensor):
        all_hidden_states = all_hidden_states.cpu().numpy()

    # Reshape hidden states to 2D array [samples, features]
    X = all_hidden_states.reshape(all_hidden_states.shape[0], -1)

    for attribute in attributes:
        try:
            logger.info(f"Probing attribute: {attribute}")

            # Extract the target variable for the current attribute
            Y = train_data[attribute].values

            logger.info(f"Feature matrix shape: {X.shape}")
            logger.info(f"Target vector shape: {Y.shape}")

            # Split the data into training and testing sets
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, random_state=42
            )

            logger.info(f"Training with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples.")

            # Initialize and train the ElasticNet model
            elastic_net = ElasticNet()
            elastic_net.fit(X_train, Y_train)

            # Make predictions on the test set
            Y_pred = elastic_net.predict(X_test)

            # Evaluate the model
            mse = mean_squared_error(Y_test, Y_pred)
            r2 = r2_score(Y_test, Y_pred)

            # Log the results
            logger.info(f"Results for attribute '{attribute}':")
            logger.info(f"  Mean Squared Error (MSE): {mse:.4f}")
            logger.info(f"  R-squared (R^2): {r2:.4f}")
            logger.info("-" * 50)
        except Exception as e:
            logger.error(f"Error while probing attribute '{attribute}': {e}")


if __name__ == '__main__':
    log = Log()
    logger = log.logger

    data_path = 'data/enVent_gen_Data.csv'
    data_encoding = 'ISO-8859-1'
    train_data = pd.read_csv(data_path, encoding=data_encoding)
    train_data['emotion'] = train_data['emotion'].map({
        "anger": 0, "boredom": 1, "disgust": 2, "fear": 3, "guilt": 4, "joy": 5,
        "no-emotion": 6, "pride": 7, "relief": 8, "sadness": 9, "shame": 10,
        "surprise": 11, "trust": 12
    }).astype(int)

    attributes = ['predict_event', 'pleasantness', 'attention', 'other_responsblt', 'chance_control', 'social_norms']
    train_data['input_text'] = train_data['hidden_emo_text'].apply(lambda x: f"{x} I feel")

    dataset = TextDataset(train_data['input_text'].tolist())
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Log model details
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded model with {num_params} parameters.")
    logger.info(f"Model configuration: {model.config}")

    try:
        logger.info("Tokenizing texts")
        logger.info("Running model inference to extract hidden states")
        all_hidden_states = process_batches(dataloader, tokenizer, model, logger)
        logger.info('hidden states saved!')
    except Exception as e:
        logger.error(f"Extracting hidden states failed: {e}")

    try:
        # Load the hidden states from file
        all_hidden_states = np.load('hidden_states.npy')
        probe(attributes, train_data, all_hidden_states, logger)
        logger.info('probe done!')
    except Exception as e:
        logger.error(f"Probing failed: {e}")
