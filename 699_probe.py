import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from sklearn.linear_model import ElasticNet, LogisticRegression,LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import logging
import os
from datetime import datetime
from tqdm import tqdm
import subprocess
import sentencepiece 
from huggingface_hub import login
from dotenv import load_dotenv
from sklearn.model_selection import cross_val_score, KFold

# Define the dataset class for handling text data
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        # self.labels = labels
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
        return text, label

# Log class to handle logging activities
class Log:
    def __init__(self):
        filename = f'logs/probe_date-{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}.txt'
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
    load_dotenv()
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
        return_tensors="pt",
        padding_side='left'  # Pad at the beginning of the sequences
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
def process_batches_multiple_files(dataloader, tokenizer, model, logger, max_length, extract_mode = 'last_token'):
    total_batches = len(dataloader)
    all_hidden_states = []
    all_labels = []  # To store labels corresponding to each batch

    for i,  (batch_texts, batch_labels) in enumerate(tqdm(dataloader, desc="Processing batches"), 1):
        if i % 20 == 0 or i == total_batches:
            logger.info(f"Completed {i}/{total_batches} batches")
        debug = True if i<2 else False    
        hidden_states = extract_hidden_states(batch_texts, tokenizer, model,  logger=logger, max_length=max_length,  extract_mode=extract_mode, debug=debug)
        # hidden_states is already a NumPy array
        all_hidden_states.append(hidden_states)  # Keep as NumPy arrays for now
        all_labels.extend(batch_labels)  # Collect labels

    # Concatenate all arrays in the list to a single NumPy array
    all_hidden_states_array = np.concatenate(all_hidden_states, axis=0)
    all_labels_array = np.array(all_labels)  # Convert list of labels to a NumPy array

    # Log the size of the hidden states array
    total_size_mb  = all_hidden_states_array.nbytes / (1024 ** 2)  # size in megabytes
    
    
    # Decide on the number of slices based on the size of the array
    if total_size_mb > 100:
        num_files = int(np.ceil(total_size_mb / 100))  # Number of files needed if each is limited to 100 MB
        slice_size = len(all_hidden_states_array) // num_files  # Number of rows per file

        for j in range(num_files):
            start_idx = j * slice_size
            end_idx = start_idx + slice_size if j != num_files - 1 else len(all_hidden_states_array)
            np.save(f'hidden_states_{j}.npy', all_hidden_states_array[start_idx:end_idx])
            logger.info(f"Saved slice {j} of hidden states as 'hidden_states_{j}.npy'")
    else:
        np.save('hidden_states.npy', all_hidden_states_array)
        logger.info("Saved all hidden states as a single .npy file")

    logger.info(f"Total size of hidden states array: {total_size_mb:.2f} MB")
    np.save('labels.npy', all_labels_array)  # Save labels in a separate file    
    return all_hidden_states_array, all_labels_array

def process_batches(dataloader, tokenizer, model, logger, max_length, extract_mode='last_token'):
    total_batches = len(dataloader)
    all_hidden_states = []
    all_labels = []  # To store labels corresponding to each batch

    for i, (batch_texts, batch_labels) in enumerate(tqdm(dataloader, desc="Processing batches"), 1):
        if i % 20 == 0 or i == total_batches:
            logger.info(f"Completed {i}/{total_batches} batches")
        debug = True if i < 2 else False
        hidden_states = extract_hidden_states(batch_texts, tokenizer, model, logger=logger, max_length=max_length, extract_mode=extract_mode, debug=debug)
        all_hidden_states.append(hidden_states)
        all_labels.extend(batch_labels)  # Collect labels

    # Concatenate all arrays in the list to a single NumPy array
    all_hidden_states_array = np.concatenate(all_hidden_states, axis=0)
    all_labels_array = np.array(all_labels)  # Convert list of labels to a NumPy array

    # Save the hidden states and labels
    np.save('hidden_states.npy', all_hidden_states_array)
    np.save('labels.npy', all_labels_array)  # Save labels in a separate file

    logger.info("Saved all hidden states and labels.")
    return all_hidden_states_array, all_labels_array

# Probing function to analyze hidden states using regression
def probe(all_hidden_states, labels, appraisals, logger):
    """
    Probes each attribute separately using regression on the hidden states.

    Args:
    - all_hidden_states: NumPy array of hidden states from the model.
    - labels: NumPy array of labels corresponding to the hidden states. First column is emotion, remaining are appraisals.
    - appraisals: List of appraisal attribute names corresponding to the columns in labels after the first column.
    - logger: Logger object for logging information.
    """
    if isinstance(all_hidden_states, torch.Tensor):
        all_hidden_states = all_hidden_states.cpu().numpy()

    X = all_hidden_states.reshape(all_hidden_states.shape[0], -1)
    Y_emotion = labels[:, 0]
    Y_appraisals = labels[:, 1:]

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Probing for emotion (classification)
    try:
        logger.info("Probing emotion category")
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target vector shape: {Y_emotion.shape}")

        cv_accuracies = cross_val_score(LogisticRegression(max_iter=2000), X, Y_emotion, cv=kfold, scoring='accuracy')
        classifier = LogisticRegression(max_iter=2000)
        classifier.fit(X, Y_emotion)  # Train on the entire dataset for full model training after CV
        training_accuracy = classifier.score(X, Y_emotion)

        logger.info(f"5-Fold CV Accuracy for emotion category: {cv_accuracies.mean():.4f} Â± {cv_accuracies.std():.4f}")
        logger.info(f"Training Accuracy for emotion category: {training_accuracy:.4f}")
    except Exception as e:
        logger.error(f"Error while probing emotion category: {e}")

    # Probing for each appraisal (regression)
    for i, appraisal_name in enumerate(appraisals):
        try:
            logger.info(f"Probing appraisal: {appraisal_name}")
            Y = Y_appraisals[:, i]
            logger.info(f"Feature matrix shape: {X.shape}")
            logger.info(f"Target vector shape: {Y.shape}")

            model = LinearRegression() # ElasticNet()
            cv_mse = cross_val_score(model, X, Y, cv=kfold, scoring='neg_mean_squared_error')
            cv_r2 = cross_val_score(model, X, Y, cv=kfold, scoring='r2')

            model.fit(X, Y)  # Train on the entire dataset for full model training after CV
            training_predictions = model.predict(X)
            training_mse = mean_squared_error(Y, training_predictions)
            training_r2 = r2_score(Y, training_predictions)

            logger.info(f"5-Fold CV MSE for '{appraisal_name}': {-cv_mse.mean():.4f} Â± {cv_mse.std():.4f}")
            logger.info(f"Training MSE for '{appraisal_name}': {training_mse:.4f}")
            logger.info(f"5-Fold CV R-squared for '{appraisal_name}': {cv_r2.mean():.4f} Â± {cv_r2.std():.4f}")
            logger.info(f"Training R-squared for '{appraisal_name}': {training_r2:.4f}")
            logger.info("- -"*25)
        except Exception as e:
            logger.error(f"Error while probing appraisal '{appraisal_name}': {e}")

def probe_classification(hidden_states, labels, appraisals, logger):
    """
    Classifies appraisal scores into low (0), medium (1), and high (2) categories, then performs logistic regression.
    
    Args:
    - hidden_states: NumPy array or torch tensor containing the hidden states from a model.
    - labels: NumPy array containing the labels for each of the hidden states. Assumed that each label set contains several appraisal scores.
    - appraisals: List of appraisal attribute names.
    - logger: Logger object to output logs.

    Returns:
    - None; logs results directly.
    """
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states.cpu().numpy()  # Convert to NumPy if tensor

    X = hidden_states.reshape(hidden_states.shape[0], -1)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Iterate over each appraisal to convert scores and perform classification
    for i, appraisal_name in enumerate(appraisals):
        # Extract scores for the current appraisal
        scores = labels[:, i]

        # Convert scores to categorical classes
        categories = np.where(scores < 3, 0, np.where(scores == 3, 1, 2))

        # Performing k-fold cross-validation and logging results
        accuracies = []
        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = categories[train_idx], categories[test_idx]

            # Initialize and train the logistic regression model
            model = LogisticRegression(max_iter=2000)
            model.fit(X_train, y_train)

            # Predict on test set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        # Log the average accuracy across the k-folds
        avg_accuracy = np.mean(accuracies)
        std_dev = np.std(accuracies)
        logger.info(f"Accuracy for {appraisal_name} (low, med, high classification): {avg_accuracy:.4f} Â± {std_dev:.4f}")
        logger.info(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

def generate_text_responses(tokenizer, model, batch_texts, max_length, num_tokens=5):
    """
    Generates text responses for a given list of input texts using the provided model and tokenizer.

    Args:
    - tokenizer: The tokenizer object.
    - model: The model used for generation.
    - batch_texts: List of text prompts.
    - max_length: Maximum length of the generated sequence.
    - num_tokens: Number of tokens to generate.

    Returns:
    - A list of generated text responses.
    """
    model.eval()  # Set the model to evaluation mode
    generated_responses = []
    first_tokens = []

    # Encode the prompts and generate responses
    for text in batch_texts:
        encoded_input = tokenizer.encode(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length,  padding_side='left', add_special_tokens=True ).to(model.device)
        outputs = model.generate(encoded_input, max_length=max_length + num_tokens, num_return_sequences=1)
        
        # Decode the generated tokens to strings
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_responses.append(generated_text)
        
        # Extract and save the first token after generation
        generated_tokens = tokenizer.convert_ids_to_tokens(outputs[0])
        if len(generated_tokens) > max_length:
            first_tokens.append(generated_tokens[max_length])  # Get the first token of the generated text

    return generated_responses, first_tokens

def compare_emotions(first_tokens, labels, logger):
    """
    Compares the first generated tokens with expected emotion labels.

    Args:
    - first_tokens: List of the first tokens from the generated texts.
    - labels: The corresponding labels for the emotions.
    - logger: Logger object for logging the comparison results.

    Returns:
    - None; logs results directly.
    """
    emotion_dict = {0: "anger", 1: "boredom", 2: "disgust", 3: "fear", 4: "guilt",
                    5: "joy", 6: "no-emotion", 7: "pride", 8: "relief", 9: "sadness",
                    10: "shame", 11: "surprise", 12: "trust"}
    
    correct_matches = 0
    total = len(first_tokens)

    for token, label in zip(first_tokens, labels):
        expected_emotion = emotion_dict.get(label, "Unknown")
        # Here you need a mapping of tokens to emotions or a way to interpret tokens as emotions
        if token in expected_emotion.lower():  # Simplistic matching; consider refining how tokens are matched to emotions
            correct_matches += 1
        logger.info(f"Expected Emotion: {expected_emotion}, Generated Token: {token}")

    accuracy = correct_matches / total if total > 0 else 0
    logger.info(f"Accuracy of matching generated tokens to expected emotions: {accuracy:.2%}")
    return accuracy

if __name__ == '__main__':
    log = Log()
    logger = log.logger
    log_system_info(logger)
    hf_login(logger)

    data_path = 'data/enVent_gen_Data.csv'
    data_encoding = 'ISO-8859-1'
    train_data = pd.read_csv(data_path, encoding=data_encoding)
    train_data = train_data[:100]

    train_data['emotion'] = train_data['emotion'].map({
        "anger": 0, "boredom": 1, "disgust": 2, "fear": 3, "guilt": 4, "joy": 5,
        "no-emotion": 6, "pride": 7, "relief": 8, "sadness": 9, "shame": 10,
        "surprise": 11, "trust": 12
    }).astype(int)

    appraisals = ['predict_event', 'pleasantness', 'attention', 'other_responsblt', 'chance_control', 'social_norms']
    train_data['input_text'] = train_data['hidden_emo_text'].apply(lambda x:f"select an emotion from the list below that matches this scenario:\n[anger, boredom, disgust, fear, guilt, joy, pride, relief, sadness, shame, surprise, trust] \n {x}. The emotion I felt was")
    labels = train_data[['emotion'] + appraisals] 

    dataset = TextDataset(train_data['input_text'].tolist(), labels)
    dataloader = DataLoader(dataset, batch_size=32)


    model_choice = 'llama3'  # Change to 'gpt2' or 'llama2', meta-llama/Llama-3.2-1B,  meta-llama/Meta-Llama-3-8B

    # Configuration for models
    if model_choice == 'gpt2':
        model_name = 'gpt2'
        tokenizer_class = AutoTokenizer
        model_class = AutoModelForCausalLM
        special_tokens = {'pad_token': '<|endoftext|>'}
        max_length = 128

    elif model_choice == 'llama2': 
        model_name = 'meta-llama/llama-2-7b-hf'  # Replace with the desired LLaMA-2 model 
        tokenizer_class = LlamaTokenizer
        model_class = LlamaForCausalLM
        special_tokens = {'pad_token': '<pad>'}
        max_length = 128  # Adjust if needed based on the model's max length
    
    elif model_choice == 'llama3': 
        model_name = 'meta-llama/Llama-3.2-1B'  # Replace with Meta-Llama-3-8B-Instruct
        tokenizer_class = AutoTokenizer
        model_class = AutoModelForCausalLM
        max_length = 128  # Adjust if needed based on the model's max length
    else:
        raise ValueError("Invalid model choice. Please select 'gpt2' or 'llama2' or 'llama3'.")
    

    # Load tokenizer and model
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, device_map="auto")

    if model_choice != 'llama3':
        # Add special tokens if necessary
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))

        # Set the padding token for the tokenizer
        tokenizer.pad_token = special_tokens['pad_token']
    else:
        # Set the padding token to the eos_token
        tokenizer.pad_token = tokenizer.eos_token

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
        _, _ = process_batches(dataloader, tokenizer, model, logger, max_length, extract_mode)
        logger.info('hidden states saved!')
    except Exception as e:
        logger.error(f"Extracting hidden states failed: {e}")

    try:
        # # Load the hidden states from file
        # all_hidden_states = np.load('hidden_states.npy')
        # labels = np.load('labels.npy')
        # # probe_classification(all_hidden_states, labels, appraisals, logger)
        # probe( all_hidden_states, labels,appraisals, logger)
        logger.info('probe done!')
    except Exception as e:
        logger.error(f"Probing failed: {e}")


    try:
        logger.info("Running text generation...")
        first_batch_texts = train_data['input_text'].tolist()[:5]  # Adjust as needed
        generated_responses, first_tokens = generate_text_responses(tokenizer, model, first_batch_texts, max_length=128, num_tokens=5)


        for i, text in enumerate(first_batch_texts):
            logger.info(f"Input: {text}")
            logger.info(f"Generated Response: {generated_responses[i]}")
            logger.info(f"First Generated Token: {first_tokens[i]}")
        
        accuracy = compare_emotions(first_tokens, labels, logger)
        logger.info(f"Final Accuracy: {accuracy:.2%}")
        
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
