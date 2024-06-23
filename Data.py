import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from Main import GEA_Emotion_Classifier

""" Dataset"""
class GEA_Dataset(Dataset):
    def __init__(self, config, data_path, emotion_or_appraisal, tokenizer, max_token_len: int, 
                use_embeddings=False):
        self.config = config
        self.data_path = data_path
        self.emotion_or_appraisal = emotion_or_appraisal
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.use_embeddings=use_embeddings
        self.emo_dict = config.emo_dict
        self._prepare_data()
        if use_embeddings:
            self.embeddings, self.input_ids, self.embed_attention_mask = self._generate_embeddings()

    def _generate_embeddings(self):
        if self.config.load_from_checkpoint:
            embed_model = GEA_Emotion_Classifier.load_from_checkpoint(self.config.checkpoint_dir, config = self.config).to(self.config.device)

        else:
            embed_model = AutoModel.from_pretrained(self.config.model_name).to(self.config.device) 

        # model.load_from_checkpoint(self.config.checkpoint_dir, map_location=self.config.device)
        embed_tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        embed_model.eval()  # Ensure the model is in evaluation mode
        texts = self.data[self.config.text].tolist()
        embeddings_list = []
        input_ids_list = []
        attention_mask_list = []
        
        for i in range(0, len(texts), self.config.val_batch_size):
            batch_texts = texts[i:i+self.config.val_batch_size]
            with torch.no_grad():  # Ensure no gradients are computed
                inputs = embed_tokenizer(batch_texts, return_tensors='pt', padding='max_length', truncation=True, 
                                        pad_to_max_length=True, max_length=self.config.max_length, return_attention_mask=True)
                # Move inputs to the same device as the model
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                if self.config.load_from_checkpoint:
                    embeddings = embed_model.extract_embeddings(**inputs)
                else:
                    outputs = embed_model(**inputs)
                    embeddings = outputs.last_hidden_state # .mean(dim=1)  # Example: mean pooling over the sequence dimension
                embeddings_list.append(embeddings.cpu())  # Move embeddings back to CPU for now
                input_ids_list.append(inputs['input_ids'].cpu())
                attention_mask_list.append(inputs['attention_mask'].cpu())
        
        # Concatenate all batches
        embeddings = torch.cat(embeddings_list, dim=0)
        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)

        return embeddings, input_ids, attention_mask
    
    def _prepare_data(self):
        data = pd.read_csv(self.data_path, encoding=self.config.data_encoding)
        self.data = data
        if self.emotion_or_appraisal == 'emotion' or self.emotion_or_appraisal == 'both':
            if 'emotion' in self.data.columns:
                self.data['emotion'] = self.data['emotion'].map(self.emo_dict).astype(int)
                self.class_num = len(self.emo_dict)
            else:
                raise ValueError(f"'emotion' column not found in {self.data_path}")      
        
        if self.emotion_or_appraisal == 'appraisal' or self.emotion_or_appraisal == 'both':
            for attr in self.config.attributes:
                if attr not in self.data.columns:
                    raise ValueError(f"'{attr}' column not found in {self.data_path}")   

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        text = str(item[self.config.text])
        if self.emotion_or_appraisal =='emotion':
            labels = torch.tensor(item[self.config.emo_attributes[0]], dtype=torch.long)
        elif self.emotion_or_appraisal =='appraisal':
            # labels = torch.tensor(item[self.config.attributes], dtype=torch.float)
            labels = torch.tensor([item[attr] for attr in self.config.attributes], dtype=torch.float)
        elif self.emotion_or_appraisal =='both':
            appraisal_labels = torch.tensor([item[attr] for attr in self.config.attributes], dtype=torch.float)
            # Collect emotion label and ensure it's the same type for concatenation
            emotion_labels = torch.tensor(item[self.config.emo_attributes[0]], dtype=torch.long)

        if self.use_embeddings:
            return {
                'inputs_embeds': self.embeddings[index].float(),
                'attention_mask': self.embed_attention_mask[index],
                'labels': labels
            }
        else:
            tokens = self.tokenizer.encode_plus(text,
                                                add_special_tokens=True, # prompt tunning
                                                return_tensors='pt',
                                                truncation=True,
                                                padding='max_length',
                                                max_length=self.max_token_len,
                                                return_attention_mask = True)
            # return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': labels}
            if not self.emotion_or_appraisal =='both':
                return {
                    'input_ids': tokens['input_ids'].squeeze(0),  # Squeeze to remove extra batch dimension added by tokenizer
                    'attention_mask': tokens['attention_mask'].squeeze(0),
                    'labels': labels
                }
            else: 
                return {
                    'input_ids': tokens['input_ids'].squeeze(0),  # Squeeze to remove extra batch dimension added by tokenizer
                    'attention_mask': tokens['attention_mask'].squeeze(0),
                    'appraisal_labels': appraisal_labels,
                    'emotion_labels': emotion_labels
                }

""" Data module"""
class GEA_Data_Module(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emotion_or_appraisal = self.config.emotion_or_appraisal
        self.train_path = self.config.train_path
        self.val_path = self.config.val_path
        self.test_path = self.config.test_path
        self.train_batch_size = self.config.train_batch_size
        self.val_batch_size = self.config.val_batch_size
        self.max_token_length = self.config.max_length
        self.model_name = self.config.model_name
        self.tokenizer = self.config.tokenizer
        self.use_input_embeddings= self.config.use_input_embeddings 
        
    def setup(self, stage = None):
        if stage in (None, "fit"):
            self.train_dataset = GEA_Dataset(self.config, self.train_path, self.emotion_or_appraisal, tokenizer= self.tokenizer, max_token_len=self.max_token_length, use_embeddings=self.use_input_embeddings)
            self.val_dataset = GEA_Dataset(self.config, self.val_path, self.emotion_or_appraisal,tokenizer= self.tokenizer,  max_token_len=self.max_token_length, use_embeddings=self.use_input_embeddings)
        if stage == 'predict':
            self.test_dataset = GEA_Dataset(self.config,self.test_path, self.emotion_or_appraisal,tokenizer= self.tokenizer,  max_token_len=self.max_token_length, use_embeddings=self.use_input_embeddings)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.train_batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.val_batch_size, num_workers=4,  shuffle=False)
        # return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4,persistent_workers=True, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.val_batch_size, num_workers=4, shuffle=False)
