import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import logging
import os
from datetime import datetime
from tqdm import tqdm
import subprocess
import sentencepiece 
from huggingface_hub import login

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

def log_system_info(logger):
    """
    Logs system memory and GPU details.
    """
    def run_command(command):
        """
        Runs a shell command and returns its output.
        
        Args:
        - command (list): Command and arguments to execute.
        
        Returns:
        - str: Output of the command.
        """
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return result.stderr
    memory_info = run_command(['free', '-h'])
    gpu_info = run_command(['nvidia-smi'])

    logger.info("Memory Info:\n" + memory_info)
    logger.info("GPU Info:\n" + gpu_info)

def hf_login(logger):
    try:
        # Retrieve the token from an environment variable
        token = os.getenv("HUGGINGFACE_TOKEN")
        if token is None:
            logger.error("Hugging Face token not set in environment variables.")
            return
        
        # Attempt to log in with the Hugging Face token
        login(token=token)
        logger.info("Logged in successfully to Hugging Face Hub.")
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")

def extract_hidden_states(batch_texts, tokenizer, model, logger, max_length, extract_mode="last_token", debug=False):
    """
    Extracts the hidden states of the last non-padded token for each input text batch.

    Args:
    - batch_texts: List of text strings to process.
    - tokenizer: Tokenizer object from the transformers library.
    - model: Pretrained model from the transformers library.
    - max_length: Maximum length of the tokenized input (default is 512).
    - logger: Logger object for logging information.
    - extract_mode: "last_token" to get the last non-padded token's hidden state, "mean" for the mean of all non-padded tokens.
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

    if extract_mode == "last_token":
        last_token_states = []
        for i in range(hidden_states.size(0)):
            index = last_token_indices[i].item()
            last_hidden_state = hidden_states[i, index, :]  # Shape: [hidden_size]
            last_token_states.append(last_hidden_state)
        
            if debug and i<2:  # Print debug information for the first two examples
                # Extract the hidden state for the last non-padded token in each sequence
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
        # Convert list to tensor and numpy array
        result_states = torch.stack(last_token_states).detach().cpu().numpy()
    
    elif extract_mode == "mean":
        mean_states = []
        for i in range(hidden_states.size(0)):
            valid_tokens = hidden_states[i, :sequence_lengths[i], :]
            mean_state = valid_tokens.mean(dim=0)
            mean_states.append(mean_state)
        
        # Convert list to tensor and numpy array
        result_states = torch.stack(mean_states).detach().cpu().numpy()

    # logger.info(f"Resultant state shape for mode '{mode}': {result_states.shape}")  # Shape: [batch_size, hidden_size]
    return result_states

# Process batches of text to get hidden states
def process_batches(dataloader, tokenizer, model, logger, max_length, extract_mode = 'last_token'):
    total_batches = len(dataloader)
    all_hidden_states = []
    for i, batch_texts in enumerate(tqdm(dataloader, desc="Processing batches"), 1):
        if i % 20 == 0 or i == total_batches:
            logger.info(f"Completed {i}/{total_batches} batches")
        debug = True if i<2 else False    
        hidden_states = extract_hidden_states(batch_texts, tokenizer, model,  logger=logger, max_length=max_length,  extract_mode=extract_mode, debug=debug)
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
def probe(appriasals, emotion, train_data, all_hidden_states, logger):
    """
    Probes each attribute separately using regression on the hidden states.

    Args:
    - appriasals: List of appriasals to probe.
    - emotion: the emotion column to probe. (categorical)
    - train_data: The DataFrame containing the target appriasals.
    - all_hidden_states: NumPy array or PyTorch tensor of hidden states.
    - logger: Logger object for logging information.
    """
    # Ensure all_hidden_states is a NumPy array
    if isinstance(all_hidden_states, torch.Tensor):
        all_hidden_states = all_hidden_states.cpu().numpy()

    # Reshape hidden states to 2D array [samples, features]
    X = all_hidden_states.reshape(all_hidden_states.shape[0], -1)

    for appriasal in appriasals:
        try:
            logger.info(f"Probing appriasal: {appriasal}")

            # Extract the target variable for the current attribute
            Y = train_data[appriasal].values

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
            logger.info(f"Results for appriasal '{appriasal}':")
            logger.info(f"  Mean Squared Error (MSE): {mse:.4f}")
            logger.info(f"  R-squared (R^2): {r2:.4f}")
            logger.info("-" * 50)
        except Exception as e:
            logger.error(f"Error while probing appriasal '{appriasal}': {e}")

    try:
        logger.info(f"Probing emotion category: {emotion}")

        Y = train_data[emotion].values
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target vector shape: {Y.shape}")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        logger.info(f"Training with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples.")

        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(Y_test, Y_pred)

        logger.info(f"Results for emotion category '{emotion}':")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("-" * 50)

    except Exception as e:
        logger.error(f"Error while probing emotion category '{emotion}': {e}")

def inspect_examples(batch_texts, tokenizer, model, max_length, num_examples=2):
    """
    Inspects tokens, embeddings, attention masks, padding, and masked attention
    for the first few examples in the provided batch_texts.

    Args:
    - batch_texts: List of text strings to process.
    - tokenizer: Tokenizer object from the transformers library.
    - model: Pretrained model from the transformers library.
    - max_length: Maximum length of the tokenized input (default is 512).
    - num_examples: Number of examples to inspect (default is 2).
    """

    # Tokenize the batch of texts
    inputs = tokenizer(
        batch_texts[:num_examples],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True
    )

    # Move inputs to the model's device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Run the model to get outputs and attentions
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )

    # Loop through each example
    for i in range(num_examples):
        logger.info(f"Example {i+1}:")
        logger.info(f"Input text: {batch_texts[i]}")

        input_ids = inputs['input_ids'][i]
        attention_mask = inputs['attention_mask'][i]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # Extract attention from the last layer
        last_layer_attention = outputs.attentions[-1][i]  # Shape: [num_heads, max_length, max_length]
        # Extract the actual sequence length (non-padded part)
        seq_len = attention_mask.sum().item()

        # Logging tokens and attention mask
        logger.info(f"Tokens: {tokens[:seq_len]}")
        logger.info(f"Token IDs: {input_ids.cpu().numpy()[:seq_len]}")
        logger.info(f"Attention Mask: {attention_mask.cpu().numpy()[:seq_len]}")
        logger.info(f"Sequence Length (excluding padding): {seq_len}")

        # Display attention weights for the first head, focusing only on the non-padded part
        head_attention = last_layer_attention[0][:seq_len, :seq_len]  # Shape: [seq_len, seq_len]
        logger.info("\nAttention weights for the last layer (first head):")
        logger.info(f"Shape of head attention: {head_attention.shape}")
        logger.info(f"Attention weights (first token attends to others): {head_attention[0]}")
        logger.info("---")

        logger.info("=" * 50)

def find_token_length_distribution(data, tokenizer):
    """
    Calculates the distribution of token lengths in the dataset, including quartiles.

    Args:
    - data: Iterable of text strings.
    - tokenizer: Tokenizer object from the transformers library.

    Returns:
    - Dictionary containing the minimum, 25th percentile, median, 75th percentile, and maximum token lengths.
    """
    token_lengths = []
    for text in data:
        tokens = tokenizer.tokenize(text)
        token_lengths.append(len(tokens))
    
    token_lengths = np.array(token_lengths)
    quartiles = np.percentile(token_lengths, [25, 50, 75])
    min_length = np.min(token_lengths)
    max_length = np.max(token_lengths)

    return {
        "min_length": min_length,
        "25th_percentile": quartiles[0],
        "median": quartiles[1],
        "75th_percentile": quartiles[2],
        "max_length": max_length
    }

if __name__ == '__main__':
    log = Log()
    logger = log.logger
    log_system_info(logger)
    data_path = 'data/enVent_gen_Data.csv'
    data_encoding = 'ISO-8859-1'
    train_data = pd.read_csv(data_path, encoding=data_encoding)
    train_data['emotion'] = train_data['emotion'].map({
        "anger": 0, "boredom": 1, "disgust": 2, "fear": 3, "guilt": 4, "joy": 5,
        "no-emotion": 6, "pride": 7, "relief": 8, "sadness": 9, "shame": 10,
        "surprise": 11, "trust": 12
    }).astype(int)

    appriasals = ['predict_event', 'pleasantness', 'attention', 'other_responsblt', 'chance_control', 'social_norms']
    train_data['input_text'] = train_data['hidden_emo_text'].apply(lambda x: f"{x}. I felt")

    dataset = TextDataset(train_data['input_text'].tolist())
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


    model_choice = 'llama2'  # Change to 'gpt2' or 'llama2'

    # Configuration for models
    if model_choice == 'gpt2':
        model_name = 'gpt2'
        tokenizer_class = AutoTokenizer
        model_class = AutoModelForCausalLM
        special_tokens = {'pad_token': '<|endoftext|>'}
        max_length = 128

    elif model_choice == 'llama2':
        model_name = 'meta-llama/Llama-2-7b-hf'  # Replace with the desired LLaMA-2 model
        tokenizer_class = LlamaTokenizer
        model_class = LlamaForCausalLM
        special_tokens = {'pad_token': '<pad>'}
        max_length = 128  # Adjust if needed based on the model's max length
    else:
        raise ValueError("Invalid model choice. Please select 'gpt2' or 'llama2'.")
    

    # Load tokenizer and model
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, device_map="auto")

    # Add special tokens if necessary
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # Set the padding token for the tokenizer
    tokenizer.pad_token = special_tokens['pad_token']

    # # Inspect the first few examples
    # first_batch_texts = train_data['input_text'].tolist()[:5]  # Adjust as needed
    # inspect_examples(first_batch_texts, tokenizer, model, max_length=512, num_examples=2)

    # # Calculate token length distribution
    # token_length_distribution = find_token_length_distribution(train_data['hidden_emo_text'], tokenizer)
    # logger.info(f"Token Length Distribution: {token_length_distribution}")

    # Log model details
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded model '{model_name}' with {num_params} parameters.")
    logger.info(f"Model configuration: {model.config}")

    extract_mode = "last_token" # or 'mean'
    try:
        logger.info("Tokenizing texts")
        logger.info("Running model inference to extract hidden states")
        all_hidden_states = process_batches(dataloader, tokenizer, model, logger, max_length, extract_mode)
        logger.info('hidden states saved!')
    except Exception as e:
        logger.error(f"Extracting hidden states failed: {e}")

    try:
        # Load the hidden states from file
        all_hidden_states = np.load('hidden_states.npy')
        probe(appriasals, 'emotion', train_data, all_hidden_states, logger)
        logger.info('probe done!')
    except Exception as e:
        logger.error(f"Probing failed: {e}")
