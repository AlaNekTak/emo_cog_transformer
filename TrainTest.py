import csv, os
import torch
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoModel
from sklearn.metrics import accuracy_score, f1_score, r2_score
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import optuna
from Config import  NoOpCallback
from Data import GEA_Data_Module, GEA_Dataset
from Model import GEA_Emotion_Classifier, MixExp_Emotion_Classifier, DoubleExp_Emotion_Classifier, ProbeEmotionClassifier
from ModelOptimizer import ModelOptimizer
from collections import defaultdict


def split_train_val(path, test_ratio, encoding, random_state):
    # Load and preprocess the data
    data = pd.read_csv('enVent_gen_Data.csv', encoding=encoding)

    # Split the dataset
    train, val = train_test_split(data, test_size=test_ratio, random_state=random_state)
    train.to_csv('enVent_gen_Data_train.csv')
    val.to_csv('enVent_gen_Data_val.csv')


def split_train_val_test(config, path_train, path_test):
    def get_majority_emotion(emotions):
        # This function returns the most common emotion or a random choice if there is a tie.
        mode_count = emotions.value_counts()
        if len(mode_count) == 0:
            return np.nan  # handle empty cases if any
        max_count = mode_count.max()
        candidates = mode_count[mode_count == max_count].index
        return np.random.choice(candidates)
    
    # Load the data
    data = pd.read_csv(path_train, encoding=config.data_encoding)
    data_test = pd.read_csv(path_test, encoding=config.data_encoding)
    
    # _, data_test = train_test_split(data_test, test_size=0.1, random_state=config.random_state)


    # Merge the hidden_emo_text from the main data to the test data based on text_id
    data_test = pd.merge(data_test, data[['text_id', 'hidden_emo_text']], on='text_id', how='left')

    # Get unique text_ids from the merged test data
    unique_text_ids = data_test['text_id'].unique()
    test = data_test[data_test['text_id'].isin(unique_text_ids)]
    
    # Define default aggregation for all other columns that do not need special handling
    default_aggregations = {col: 'first' for col in test.columns if col not in config.attributes and col not in ['hidden_emo_text', 'emotion', 'annotator_emotion']}

    # Aggregate appraisal scores by averaging and get the majority for the emotion column
    aggregation_dict = {appraisal: 'mean' for appraisal in config.attributes}
    aggregation_dict['hidden_emo_text'] = 'first'  # Assuming text is identical for the same text_id
    aggregation_dict['emotion'] = 'first'  # Preserve the original emotion label from the main data
    aggregation_dict['annotator_emotion'] = lambda x: get_majority_emotion(x)
    aggregation_dict.update(default_aggregations)  # Include default aggregations

    # aggregation_dict['hidden_emo_text','emotion'] = 'first'  # Assuming text is identical for the same text_id

    test = test.groupby('text_id', as_index=False).agg(aggregation_dict).reset_index()
    
    # In case an unwanted 'index' column appears
    test = test.drop(columns=['index'], errors='ignore')



    # Ensure no overlap of text_ids between the test set and remaining training data
    train_val = data[~data['text_id'].isin(unique_text_ids)]
    
    # Final split for train and validation sets
    train, val = train_test_split(train_val, test_size=config.train_val_split, random_state=config.random_state)
    
    # Save the datasets to CSV files
    train.to_csv('enVent_new_Data_train.csv', index=False)
    val.to_csv('enVent_new_Data_val.csv', index=False)
    test.to_csv('enVent_new_Data_test.csv', index=False)
    return train, val, test


def inspect_data(config, logger):
    logger.info(config.train_data.head(5))
    logger.info(config.val_data.head(6))
    logger.info(f'len(train_data): \n {len(config.train_data)}')
    logger.info(f'len(val_data): \n {len(config.val_data)}')
    attributes = config.attributes
    class_num = config.class_num # num of unique labels
    logger.info(f'class_num: \n {class_num}')
    # train_data[attributes].plot.bar()
    # plt.show()


def quick_test(config, logger):
    model = GEA_Emotion_Classifier(config)
    if not config.use_input_embeddings:
        GEA_ds = GEA_Dataset(config, config.train_path,config.emotion_or_appraisal, config.tokenizer, config.max_length)
        logger.info(f'GEA_ds.__getitem__(0):\n {GEA_ds.__getitem__(0)}')
        logger.info(f"GEA_ds.__getitem__(0)['labels'].shape, GEA_ds.__getitem__(0)['input_ids'].shape, GEA_ds.__getitem__(0)['attention_mask'].shape \n', {GEA_ds.__getitem__(0)['labels'].shape}, {GEA_ds.__getitem__(0)['input_ids'].shape}, {GEA_ds.__getitem__(0)['attention_mask'].shape}")
        GEA_data_module = GEA_Data_Module(config)
        GEA_data_module.setup()
        GEA_data_module.train_dataloader()
        logger.info(f"len(GEA_data_module.train_dataloader()): {len(GEA_data_module.train_dataloader())}")

        idx=2
        input_ids = GEA_ds.__getitem__(idx)['input_ids']
        attention_mask = GEA_ds.__getitem__(idx)['attention_mask']
        labels = GEA_ds.__getitem__(idx)['labels']
        logger.info(f"labels: \n {labels}")

        model.cpu()
        loss, logits = model(input_ids=input_ids.unsqueeze(dim=0), attention_mask=attention_mask.unsqueeze(dim=0), labels=labels.unsqueeze(dim=0))
        logger.info(f"labels.unsqueeze(dim=0): ', {labels.unsqueeze(dim=0)},'\nlogits: ', {logits},'\nlogits shape: ', {logits.shape}")
        # loss, log_probs = model(input_ids=input_ids.unsqueeze(dim=0), attention_mask=attention_mask.unsqueeze(dim=0), labels=labels.unsqueeze(dim=0))
        # print('labels.unsqueeze(dim=0): ', labels.unsqueeze(dim=0),'\nlog_probs: ', log_probs,'\nlog_probs: ', log_probs.shape)

    elif config.use_input_embeddings:
        """ Input embedds """
        ## Load the checkpoint
        # checkpoint = torch.load('checkpoint_emotion.pth.tar', map_location=device)
        # checkpoint = torch.load('checkpoint_appraisal.pth.tar', map_location=device)
        embed_model = AutoModel.from_pretrained(config.model_name)
        ## Update the model's state dictionary
        # embed_model.load_state_dict(checkpoint['state_dict'], strict=False)
        embed_model.to(config.device)

        GEA_ds_embed = GEA_Dataset(config, config.train_path, config.emotion_or_appraisal, config.tokenizer,  config.max_length, use_embeddings=True)
        logger.info(f'"GEA_ds_embed.__getitem__(0): \n", {GEA_ds_embed.__getitem__(0)}')
        logger.info(f"# GEA_ds_embed.__getitem__(0)['labels'].shape, GEA_ds_embed.__getitem__(0)['inputs_embeds'].shape, GEA_ds_embed.__getitem__(0)['attention_mask'].shape: \n, {GEA_ds_embed.__getitem__(0)['labels'].shape}, {GEA_ds_embed.__getitem__(0)['inputs_embeds'].shape}, {GEA_ds_embed.__getitem__(0)['attention_mask'].shape}")

        GEA_data_module_embed = GEA_Data_Module(config)
        GEA_data_module_embed.setup()
        GEA_data_module_embed.train_dataloader()
        logger.info(f"len(GEA_data_module_embed.train_dataloader()): \n {len(GEA_data_module_embed.train_dataloader())}")
        idx=2
        inputs_embeds = GEA_ds_embed.__getitem__(idx)['inputs_embeds']
        attention_mask = GEA_ds_embed.__getitem__(idx)['attention_mask']
        labels = GEA_ds_embed.__getitem__(idx)['labels']
        logger.info(f"inputs_embeds: \n {inputs_embeds}")

        loss, logits = model( inputs_embeds=inputs_embeds.unsqueeze(dim=0), attention_mask=attention_mask.unsqueeze(dim=0), labels= labels.unsqueeze(dim=0))
        logger.info(f"labels.unsqueeze(dim=0): ', {labels.unsqueeze(dim=0)},'\nlogits: ', {logits},'\nlogits shape: ', {logits.shape}")


def test(model_path ,config, logger):
    """ Test """
    def evaluate_regression_model(config, model, dm, trainer, val_data, is_distributed=False):
        logger.info("\n\nStarting prediction model...\n---------------------------------")
        predict_results = trainer.predict(model, datamodule=dm)
        logits = []
        labels = []
        # Extract logits and labels from the results
        for batch_result in predict_results:
            logits.append(batch_result["predictions"])
            labels.append(batch_result["labels"])
        
        logger.info(f"Collected {len(logits)} batches from predict.")

        # Concatenate all batch logits into a single tensor along the batch dimension
        logits = torch.cat([batch for batch in logits], dim=0).to(config.device)
        labels = torch.cat([batch for batch in labels], dim=0).to(config.device)


        if not is_distributed: #torch.distributed.get_rank() == 0 or 
            predictions_np = logits.cpu().numpy()
            true_labels_np = labels.cpu().numpy()   
    
            overall_r2  = r2_score(true_labels_np, predictions_np)
            logger.info(f"R2 Score: {overall_r2 :.4f}")
            
            # Report R2 score for each attribute
            for idx, col in enumerate(config.attributes):
                attribute_predictions = predictions_np[:, idx]
                attribute_labels = true_labels_np[:, idx]
                attribute_r2 = r2_score(attribute_labels, attribute_predictions)
                logger.info(f"R2 Score for {col}: {attribute_r2:.4f}")

            for idx, col in enumerate(config.attributes):
                val_data[f"predicted_{col}"] = predictions_np[:, idx]

            return val_data

    def classify_raw_comments(model, dm, trainer):
        logger.info("\n\nStarting prediction model...\n---------------------------------")
        # Use trainer.predict to get the logits from the model; predict returns an iterable over batches
        predict_results = trainer.predict(model, datamodule=dm)
        logits = []
        labels = []
        
        # Extract logits and labels from the results
        for batch_result in predict_results:
            logits.append(batch_result["predictions"])
            labels.append(batch_result["labels"])

        logger.info(f"Collected {len(logits)} batches from predict.")

        # Concatenate all batch logits into a single tensor along the batch dimension
        logits = torch.cat([batch for batch in logits], dim=0)
        labels = torch.cat([batch for batch in labels], dim=0)
        # logger.info(f"Concatenated logits tensor shape: {logits.shape} with rank: {dist.get_rank()}")

        # Apply softmax to convert logits to probabilities along the class dimension
        probabilities = F.softmax(logits, dim=1)
        logger.info(f"Probabilities tensor shape after softmax: {probabilities.shape}")
        return probabilities, logits, labels

    def evaluate_Classification_model(config, val_data, probs, true_labels, is_distributed=False): # or log_probs_list  
        # Ensure true_labels is a PyTorch tensor instead of pandas series
        if not isinstance(true_labels, torch.Tensor):
            true_labels = torch.tensor(true_labels)
            # now we can send it to device (GPU)
        probs = probs.to(config.device)
        true_labels = true_labels.to(config.device)

        # if is_distributed:
        #     print(f"Rank {dist.get_rank()}: Gathering data...")
        #     # Assuming logits and true_labels are tensors on the appropriate device
        #     # Gather all logits and labels to the GPU with rank 0
        #     gathered_probs = [torch.zeros_like(probs) for _ in range(dist.get_world_size())]
        #     gathered_labels = [torch.zeros_like(true_labels) for _ in range(dist.get_world_size())]

        #     dist.all_gather(gathered_probs, probs)
        #     dist.all_gather(gathered_labels, true_labels)
        #     print('torch.distributed.get_rank(): ', torch.distributed.get_rank())
        #     print('before gathering on master rank \nprobs shape: ', probs.shape)
        #     if torch.distributed.get_rank() == 0:
        #         probs = torch.cat(gathered_probs, dim=0)
        #         true_labels = torch.cat(gathered_labels, dim=0)
        #         print("Data gathered on master rank.")
        #     else:
        #         print(f"Rank {dist.get_rank()}: Exiting as non-master.")
        #         return None  # Other GPUs do nothing further
        #     print('after gathering on master rank \nprobs shape: ', probs.shape)

        if not is_distributed: #torch.distributed.get_rank() == 0 or 
            predictions = torch.max(probs, 1)[1]
            predictions_np = predictions.cpu().numpy()
            true_labels_np = true_labels.cpu().numpy()   
    
            # Calculate Accuracy
            accuracy = accuracy_score(true_labels_np, predictions_np)
            logger.info(f"Accuracy: {accuracy:.4f}")
            
            # Calculate AUC-ROC (binary classification)
            if probs.shape[1] == 2:  # Binary classification check
                # Assuming the positive class probability is the second column
                roc_auc = roc_auc_score(true_labels_np, probs[:, 1].cpu().numpy())
                logger.info(f"AUC ROC: {roc_auc:.4f}")
            else:
                # Multi-class AUC-ROC (assuming one-vs-rest calculation)
                logger.info(f'true_labels_np: {true_labels_np.shape}')
                logger.info(f'probs:  {probs.cpu().numpy().shape}')
                roc_auc = roc_auc_score(true_labels_np , probs.cpu().numpy(), multi_class='ovr')
                logger.info(f"AUC ROC (One-vs-Rest): {roc_auc:.4f}")
            
            # Calculate F1 Score
            f1 = f1_score(true_labels_np, predictions_np, average='weighted')
            logger.info(f"F1 Score: {f1:.4f}")
            
            # Append predictions to the original validation DataFrame
            val_data['predictions'] = predictions_np
            
            return val_data
    
    def evalute_both(config, model, dm, trainer, val_data, is_distributed=False):
        logger.info("\n\nStarting prediction model...\n---------------------------------")
        predict_results = trainer.predict(model, datamodule=dm)
        emotion_logits, appraisal_logits, emotion_labels, appraisal_labels, gate_weights = [], [], [], [], []

        try:
            # Extract logits and labels from the results
            for idx, batch_result in enumerate(predict_results):
                emotion_logits.append(batch_result["emotion_logits"])
                emotion_labels.append(batch_result["emotion_labels"])
                appraisal_labels.append(batch_result["appraisal_labels"])
                gate_weights.append(batch_result["gate_weights"])
                
                if config.expert_mode == 'double':
                    appraisal_logits.append(batch_result["appraisal_logits"].view(-1, 1))

                elif config.expert_mode == 'mixed':
                    # Prepare to reshape appraisal logits
                    batch_appraisal_logits = [batch_result["appraisal_logits"][i] for i in range(len(batch_result["appraisal_logits"]))]
                    batch_appraisal_logits = torch.cat(batch_appraisal_logits, dim=1)  # Should reshape each batch's logits to [32, 7]
                    appraisal_logits.append(batch_appraisal_logits)
                    
            logger.info(f"Collected {len(emotion_logits)} emotion logit batches from predict.")
            logger.info(f"Collected {len(appraisal_logits)} appriasal label batches from predict.")
            logger.info(f"emotion logit shape {emotion_logits[0].shape} ")
            logger.info(f"appraisal label shape {appraisal_logits[0].shape} ")

            # Concatenate all batches into a single tensor along the batch dimension
            emotion_logits = torch.cat(emotion_logits, dim=0).to(config.device)
            emotion_labels = torch.cat(emotion_labels, dim=0).to(config.device)
            appraisal_labels = torch.cat(appraisal_labels, dim=0).to(config.device)

            gate_weights = torch.cat(gate_weights, dim=0).to(config.device)       
            appraisal_logits = torch.cat(appraisal_logits, dim=0).to(config.device)

        except RuntimeError as e:
            logger.error("Error during tensor concatenation: " + str(e))
            raise

        if not is_distributed: #torch.distributed.get_rank() == 0 or 
            predictions_np = appraisal_logits.cpu().numpy()
            true_labels_np = appraisal_labels.cpu().numpy()   
            gate_weights_np = gate_weights.cpu().numpy()
    
            overall_r2  = r2_score(true_labels_np, predictions_np)
            logger.info(f"R2 Score: {overall_r2 :.4f}")
            
            # Report R2 score for each attribute
            for idx, col in enumerate(config.attributes):
                attribute_predictions = predictions_np[:, idx]
                attribute_labels = true_labels_np[:, idx]
                attribute_r2 = r2_score(attribute_labels, attribute_predictions)
                logger.info(f"R2 Score for {col}: {attribute_r2:.4f}")

            for idx, col in enumerate(config.attributes):
                val_data[f"predicted_{col}"] = predictions_np[:, idx]
                val_data[f'gate_weights_{col}']= gate_weights_np[:, idx]

            # Emotion
                
            probs = F.softmax(emotion_logits, dim=1)
            # # Ensure true_labels is a PyTorch tensor instead of pandas series
            # if not isinstance(emotion_labels, torch.Tensor):
            #     emotion_labels = torch.tensor(emotion_labels)

            emo_predictions = torch.max(probs, 1)[1]
            emo_predictions_np = emo_predictions.cpu().numpy()
            emo_true_labels_np = emotion_labels.cpu().numpy()   
    
            # Calculate Accuracy
            accuracy = accuracy_score(emo_true_labels_np, emo_predictions_np)
            logger.info(f"Accuracy: {accuracy:.4f}")
            
            # Calculate AUC-ROC (binary classification)
            if probs.shape[1] == 2:  # Binary classification check
                # Assuming the positive class probability is the second column
                roc_auc = roc_auc_score(emo_true_labels_np, probs[:, 1].cpu().numpy())
                logger.info(f"AUC ROC: {roc_auc:.4f}")
            else:
                # Multi-class AUC-ROC (assuming one-vs-rest calculation)
                logger.info(f'emo_true_labels_np: {emo_true_labels_np.shape}')
                logger.info(f'probs:  {probs.cpu().numpy().shape}')
                roc_auc = roc_auc_score(emo_true_labels_np , probs.cpu().numpy(), multi_class='ovr')
                logger.info(f"AUC ROC (One-vs-Rest): {roc_auc:.4f}")
            
            # Calculate F1 Score
            f1 = f1_score(emo_true_labels_np, emo_predictions_np, average='weighted')
            logger.info(f"F1 Score: {f1:.4f}")

            reverse_emo_dict = {v: k for k, v in config.emo_dict.items()}
            emo_labels_np = np.array([reverse_emo_dict[idx] for idx in emo_predictions_np])
            
            # Append predictions to the original validation DataFrame
            val_data['emo_predictions'] = emo_labels_np

        return val_data

    def sptoken_emo_evaluation(config, model, dm, trainer, val_data, is_distributed=False):
        logger.info("\n\nStarting prediction model...\n---------------------------------")
        predict_results = trainer.predict(model, datamodule=dm)
        emotion_logits, emotion_labels, appraisal_labels, mean_last_hidden, attention = [], [], [], [], []

        # Reverse mappings for emotions
        reverse_emo_dict = {v: k for k, v in config.emo_dict.items()}

        try:

            # Extract logits and labels from the results
            for idx, batch_result in enumerate(predict_results):
                emotion_logits.append(batch_result["emotion_logits"])
                emotion_labels.append(batch_result["emotion_labels"])
                appraisal_labels.append(batch_result["appraisal_labels"])

                # Focus on [CLS] token attentions to and from last 7 tokens
                current_attention = batch_result["attention"]
                cls_to_appraisals = current_attention[:, :, 0, -config.n_attributes:]  # Shape: [num_heads, 1, 7]
                appraisals_to_cls = current_attention[:, :, -config.n_attributes:, 0]  # Shape: [num_heads, 7, 1]
                    
                # Average over heads
                cls_to_appraisals_avg = cls_to_appraisals.mean(axis=0)
                appraisals_to_cls_avg = appraisals_to_cls.mean(axis=0)
                attention.append((cls_to_appraisals_avg, appraisals_to_cls_avg))

            logger.info(f"Collected {len(emotion_logits)} emotion logit batches from predict.")
            logger.info(f"emotion logit shape {emotion_logits[0].shape} ")

            # Concatenate all batches into a single tensor along the batch dimension
            emotion_logits = torch.cat(emotion_logits, dim=0).to(config.device)
            emotion_labels = torch.cat(emotion_labels, dim=0).to(config.device)
            appraisal_labels = torch.cat(appraisal_labels, dim=0).to(config.device)

            # Organize attention by emotion labels
            attention_by_label = defaultdict(list)
            for logits, labels, att in zip(emotion_logits, emotion_labels, attention):
                label = labels.item()  # Assuming single label per batch
                attention_by_label[label].append(att)

            # Compute average attention per label
            average_attention_by_label = {}
            for label, att_list in attention_by_label.items():
                # Average over all batches corresponding to each label
                cls_to_appraisals_avg = np.mean([att[0] for att in att_list], axis=0)
                appraisals_to_cls_avg = np.mean([att[1] for att in att_list], axis=0)
                average_attention_by_label[label] = (cls_to_appraisals_avg, appraisals_to_cls_avg)
                
            # Calculate mean attention and organize it into a DataFrame for CSV output
            # Prepare data for CSV output with annotated emotion labels
            data_for_csv = {
                'emotion_label': [reverse_emo_dict[label] for label in average_attention_by_label],  # Annotate with emotion names
                'cls_to_appraisals_avg': [att[0].tolist() for att in average_attention_by_label.values()],
                'appraisals_to_cls_avg': [att[1].tolist() for att in average_attention_by_label.values()]
            }


            # Validate attention shapes
            if attention:
                sample_attention = attention[0]
                logger.info(f"Sample cls_to_appraisals attention shape: {sample_attention[0].shape}")
                logger.info(f"Sample appraisals_to_cls attention shape: {sample_attention[1].shape}")


            # Create a DataFrame
            df = pd.DataFrame(data_for_csv)
            
            # Save the DataFrame to CSV
            csv_path = 'output/cls_to_appraisal_attention_data.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"Attention data saved to CSV at: {csv_path}")

        except RuntimeError as e:
            logger.error("Error during tensor concatenation: " + str(e))
            raise

        if not is_distributed: #torch.distributed.get_rank() == 0 or 
            probs = F.softmax(emotion_logits, dim=1)
            emo_predictions = torch.max(probs, 1)[1]
            emo_predictions_np = emo_predictions.cpu().numpy()
            emo_true_labels_np = emotion_labels.cpu().numpy()   
    
            # Calculate Accuracy
            accuracy = accuracy_score(emo_true_labels_np, emo_predictions_np)
            logger.info(f"Accuracy: {accuracy:.4f}")
            
            # Calculate AUC-ROC (binary classification)
            if probs.shape[1] == 2:  # Binary classification check
                # Assuming the positive class probability is the second column
                roc_auc = roc_auc_score(emo_true_labels_np, probs[:, 1].cpu().numpy())
                logger.info(f"AUC ROC: {roc_auc:.4f}")
            else:
                # Multi-class AUC-ROC (assuming one-vs-rest calculation)
                logger.info(f'emo_true_labels_np: {emo_true_labels_np.shape}')
                logger.info(f'probs:  {probs.cpu().numpy().shape}')
                roc_auc = roc_auc_score(emo_true_labels_np , probs.cpu().numpy(), multi_class='ovr')
                logger.info(f"AUC ROC (One-vs-Rest): {roc_auc:.4f}")
            
            # Calculate F1 Score
            f1 = f1_score(emo_true_labels_np, emo_predictions_np, average='weighted')
            logger.info(f"F1 Score: {f1:.4f}")

            # reverse_emo_dict = {v: k for k, v in config.emo_dict.items()}
            emo_labels_np = np.array([reverse_emo_dict[idx] for idx in emo_predictions_np])
            
            # Append predictions to the original validation DataFrame
            val_data['emo_predictions'] = emo_labels_np

        return val_data    
        
    if config.emotion_or_appraisal == 'both':
        if config.expert_mode == 'double':
            model = DoubleExp_Emotion_Classifier.load_from_checkpoint(model_path, config = config).to(config.device)
        elif config.expert_mode == 'mixed':
            model = MixExp_Emotion_Classifier.load_from_checkpoint(model_path, config = config).to(config.device)
        elif config.expert_mode == 'probe' or config.expert_mode == 'sptoken':
            # model = ProbeEmotionClassifier.load_from_checkpoint(model_path, config = config).to(config.device)
            model = AutoModel.from_pretrained(config.model_name).to(config.device)
    else:
        model = GEA_Emotion_Classifier.load_from_checkpoint(model_path, config = config).to(config.device)
    # checkpoint = torch.load(model_path) # map_location=torch.device(device)
    # logger.info(f"checkpoint['keys']:\n {checkpoint.keys()}")
    # logger.info(f"Hyperparameters used: {checkpoint['hyper_parameters']}")
    trainer = pl.Trainer(max_epochs=1, 
                        accelerator="auto",
                        callbacks=[NoOpCallback()],
                        # log_every_n_steps=1, 
                        devices=1,
                        # logger = logger
                        )
    GEA_data_module = GEA_Data_Module(config)
    GEA_data_module.setup()
    is_distributed =  trainer.world_size > 1
    logger.info(f'is_distributed:  {is_distributed}')
    # dist.barrier()
    test_data = pd.read_csv(config.test_path, encoding=config.data_encoding)
   
    if config.emotion_or_appraisal == 'emotion':
        probs, logits, labels= classify_raw_comments(model, GEA_data_module, trainer)
        if trainer.global_rank == 0:
            val_data_ = evaluate_Classification_model(config, test_data, probs, labels, is_distributed)

    elif config.emotion_or_appraisal == 'appraisal':
            val_data_ = evaluate_regression_model(config, model, GEA_data_module, trainer, test_data, is_distributed)

    elif config.emotion_or_appraisal == 'both':
            if config.expert_mode == 'probe':
                do_probe(config,logger, model, GEA_data_module, trainer, test_data, is_distributed)
            elif config.expert_mode == 'sptoken':
                val_data_ = sptoken_emo_evaluation(config, model, GEA_data_module, trainer, test_data, is_distributed)
            else:
                val_data_ = evalute_both(config, model, GEA_data_module, trainer, test_data, is_distributed)
            

    if config.expert_mode != 'probe':
        val_data_.to_csv(os.path.join(config.log_path,
                                      f'data_out_mode-{config.mode}_type-{config.emotion_or_appraisal}_date-{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}.csv'))


def train(config, logger):

    optimizer = ModelOptimizer(config, logger)
    study = optuna.create_study(study_name="MyOptimizationStudy", direction="minimize")
    study.optimize(optimizer.objective, n_trials=config.hparam_trials)

    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value:.2f}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    best_model_path = optimizer.checkpoint_callback.best_model_path
    logger.info(f"Best model saved at: {best_model_path}")
    return best_model_path


def do_probe(config, logger, model, dm, trainer, test_data, is_distributed=False):
    logger.info("\n\nStarting probe model...\n---------------------------------")
    predict_results = trainer.predict(model, datamodule=dm)
    emotion_logits, emotion_labels, appraisal_labels, mean_last_hidden, attentions = [], [], [], [], []

    try:
        for idx, batch_result in enumerate(predict_results):
            emotion_logits.append(batch_result["emotion_logits"])
            emotion_labels.append(batch_result["emotion_labels"])
            appraisal_labels.append(batch_result["appraisal_labels"])
            mean_last_hidden.append(batch_result["mean_last_hidden"])
        
        logger.info(f"Collected {len(emotion_logits)} batches of emotion logits.")
        logger.info(f"emotion logits shape: {emotion_logits[0].shape}")

        emotion_logits = torch.cat(emotion_logits, dim=0).to(config.device)
        emotion_labels = torch.cat(emotion_labels, dim=0).to(config.device)
        appraisal_labels = torch.cat(appraisal_labels, dim=0).to(config.device)
        mean_last_hidden = torch.cat(mean_last_hidden, dim=0).to(config.device)

    except RuntimeError as e:
        logger.error("Error during tensor concatenation: " + str(e))
        raise

    if not is_distributed:
        # Save the hidden states and labels for later use
        output_dir = config.output_dir if hasattr(config, 'output_dir') else './output'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hidden_file_path = os.path.join(output_dir, f'hidden_states_{timestamp}.pt')
        labels_file_path = os.path.join(output_dir, f'appraisal_labels_{timestamp}.pt')

        torch.save(mean_last_hidden, hidden_file_path)
        torch.save(appraisal_labels, labels_file_path)

        logger.info(f"Saved hidden states to {hidden_file_path}")
        logger.info(f"Saved appraisal labels to {labels_file_path}")

        # Optionally save emotion logits and labels if needed
        # torch.save(emotion_logits, os.path.join(output_dir, f'emotion_logits_{timestamp}.pt'))
        # torch.save(emotion_labels, os.path.join(output_dir, f'emotion_labels_{timestamp}.pt'))

    logger.info(f"Probing complete. Data saved for further analysis.")

