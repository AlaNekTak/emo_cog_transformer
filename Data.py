import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from Model import GEA_Emotion_Classifier

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
                inputs = embed_tokenizer(batch_texts,
                                        return_tensors='pt', 
                                        padding='max_length', 
                                        truncation=True, 
                                        # pad_to_max_length=True, 
                                        max_length=self.config.max_length, 
                                        return_attention_mask=True)
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
        if self.emotion_or_appraisal in ['emotion' ,'both']:
            if 'emotion' in self.data.columns:
                self.data['emotion'] = self.data['emotion'].map(self.emo_dict).astype(int)
                self.class_num = len(self.emo_dict)
            else:
                raise ValueError(f"'emotion' column not found in {self.data_path}")      
        
        if self.emotion_or_appraisal in ['appraisal' ,'both']:
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
            

class SPToken_Dataset(Dataset):
    '''
    This class adds appraisal labels as special tokens to the input text
    
    '''
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
        if self.config.load_from_checkpoint:
            self.embed_model = GEA_Emotion_Classifier.load_from_checkpoint(self.config.checkpoint_dir, 
                                                                      config = self.config).to(self.config.device)
        else:
            self.embed_model = AutoModel.from_pretrained(self.config.model_name).to(self.config.device) 
        self.hidden_size = self.embed_model.config.hidden_size 

        if '[APP_SCORE]' not in self.tokenizer.additional_special_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['[APP_SCORE]']})
            self.embed_model.resize_token_embeddings(len(self.tokenizer))
            
        if use_embeddings:
            self.embeddings, self.input_ids, self.embed_attention_mask = self._generate_embeddings()


        # # Define and add special tokens for each appraisal attribute
        # self.special_tokens = [f'[{attr.upper()}]' for attr in self.config.attributes]
        # self.tokenizer.add_special_tokens({'additional_special_tokens': self.special_tokens})
            
    def _generate_embeddings(self):
        # model.load_from_checkpoint(self.config.checkpoint_dir, map_location=self.config.device)
        embed_tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.embed_model.eval()  # Ensure the model is in evaluation mode
        texts = self.data[self.config.text].tolist()
        texts = [f'{text} [APP_SCORE]' for text in texts] # seems space between text adn appscore is necessary here
        embeddings_list = []
        input_ids_list = []
        attention_mask_list = []
        
        for i in range(0, len(texts), self.config.val_batch_size):
            batch_texts = texts[i:i+self.config.val_batch_size]
            with torch.no_grad():  # Ensure no gradients are computed
                inputs = embed_tokenizer(batch_texts,
                                        return_tensors='pt', 
                                        padding='max_length', 
                                        truncation=True, 
                                        # pad_to_max_length=True, 
                                        add_special_tokens=True,
                                        max_length=self.max_token_len-self.config.n_attributes, 
                                        return_attention_mask=True)
                # Move inputs to the same device as the model
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                if self.config.load_from_checkpoint:
                    embeddings = self.embed_model.extract_embeddings(**inputs)
                else:
                    # outputs = embed_model(**inputs)
                    # embeddings = outputs.last_hidden_state # .mean(dim=1)  # Example: mean pooling over the sequence dimension
                    embeddings = self.embed_model.embeddings(inputs['input_ids'])
                    raw_token_embeddings = self.embed_model.embeddings.word_embeddings(inputs['input_ids'])
                    # positional_embeddings = self.embed_model.embeddings.position_embeddings(torch.arange(0, self.max_token_len, device=inputs['input_ids'].device))

                embeddings_list.append(embeddings.cpu())  # Move embeddings back to CPU for now
                input_ids_list.append(inputs['input_ids'].cpu())
                attention_mask_list.append(inputs['attention_mask'].cpu())
        
        # Concatenate all batches
        embeddings = torch.cat(embeddings_list, dim=0)
        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)

        return embeddings, input_ids, attention_mask

    def __getitem__(self, index):
        item = self.data.iloc[index]
        text = str(item[self.config.text])
        appraisal_labels = torch.tensor([item[attr] for attr in self.config.attributes], dtype=torch.float)
        emotion_labels = torch.tensor(item[self.config.emo_attributes[0]], dtype=torch.long)

        # Asserting shapes and data types
        assert isinstance(text, str), "Text must be a string"
        assert appraisal_labels.ndim == 1, "Appraisal labels must be a 1D tensor with the shape [number_of_attributes]"
        assert emotion_labels.dim() == 0, "Emotion labels should be a scalar tensor"

        if self.use_embeddings:
            # app_token_id = self.tokenizer.convert_tokens_to_ids('[APP_SCORE]')
            # app_token_embedding = self.embed_model.embeddings(torch.tensor([[app_token_id]], device=self.config.device))

            return {
                'inputs_embeds': self.embeddings[index].float(),
                'attention_mask': self.embed_attention_mask[index],
                'appraisal_labels': appraisal_labels,
                'emotion_labels': emotion_labels
            }
        else:
            appraisal_text = ','.join([f'{score}' for score in appraisal_labels])
            text = f'[APP_SCORE]{appraisal_text}[SEP]{text}'

            tokens = self.tokenizer.encode_plus(text,
                                                add_special_tokens=True, # prompt tunning
                                                return_tensors='pt',
                                                truncation=True,
                                                padding='max_length',
                                                max_length=self.max_token_len,
                                                return_attention_mask = True)
            
            assert tokens['input_ids'].size(1) <= self.max_token_len, "Token length exceeds maximum"
            return {
                'input_ids': tokens['input_ids'].squeeze(0),  # Squeeze to remove extra batch dimension added by tokenizer
                'attention_mask': tokens['attention_mask'].squeeze(0),
                'appraisal_labels': appraisal_labels,
                'emotion_labels': emotion_labels
            }
        
    def __getitem2__(self, index):
        item = self.data.iloc[index]
        text = str(item[self.config.text])
        appraisal_labels = torch.tensor([item[attr] for attr in self.config.attributes], dtype=torch.float)
        emotion_labels = torch.tensor(item[self.config.emo_attributes[0]], dtype=torch.long)

        # Asserting shapes and data types
        assert isinstance(text, str), "Text must be a string"
        assert appraisal_labels.ndim == 1, "Appraisal labels must be a 1D tensor with the shape [number_of_attributes]"
        assert emotion_labels.dim() == 0, "Emotion labels should be a scalar tensor"
        
        if self.use_embeddings:
            self.embed_model.eval()  # Ensure the model is in evaluation mode
            with torch.no_grad():  # Ensure no gradients are computed
                inputs = self.tokenizer.encode_plus(text, 
                                        return_tensors='pt', 
                                        padding='max_length', 
                                        truncation=True, 
                                        max_length=self.max_token_len-self.config.n_attributes, 
                                        return_attention_mask=True
                                        )
                # Move inputs to the same device as the model
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                if self.config.load_from_checkpoint:
                    embeddings = self.embed_model.extract_embeddings(**inputs)
                else:
                    # outputs = self.embed_model(**inputs)
                    # embeddings = outputs.last_hidden_state # .mean(dim=1)  # Example: mean pooling over the sequence dimension
                    embeddings = self.embed_model.embeddings(inputs['input_ids'])
                input_ids=inputs['input_ids']
                attention_mask = inputs['attention_mask']
            
            assert embeddings.size(0) == 1, "Embeddings batch size should be 1 [1, seq_length, hidden_size]"
            assert embeddings.size(-1) == self.hidden_size, "Embeddings size mismatch"

        if self.use_embeddings:
            return {
                'inputs_embeds': embeddings.squeeze(0),
                'attention_mask': attention_mask.squeeze(0),
                'appraisal_labels': appraisal_labels,
                'emotion_labels': emotion_labels
            }
        else:
            appraisal_text = ','.join([f'{score}' for score in appraisal_labels])
            text = f'[APP_SCORE]{appraisal_text}[SEP]{text}'

            tokens = self.tokenizer.encode_plus(text,
                                                add_special_tokens=True, # prompt tunning
                                                return_tensors='pt',
                                                truncation=True,
                                                padding='max_length',
                                                max_length=self.max_token_len,
                                                return_attention_mask = True)
            
            assert tokens['input_ids'].size(1) <= self.max_token_len, "Token length exceeds maximum"

            return {
                'input_ids': tokens['input_ids'].squeeze(0),  # Squeeze to remove extra batch dimension added by tokenizer
                'attention_mask': tokens['attention_mask'].squeeze(0),
                'appraisal_labels': appraisal_labels,
                'emotion_labels': emotion_labels
            }
    
    def _prepare_data(self):
        self.data = pd.read_csv(self.data_path, encoding=self.config.data_encoding)
        if 'emotion' in self.data.columns:
            self.data['emotion'] = self.data['emotion'].map(self.emo_dict).astype(int)
            self.class_num = len(self.emo_dict)
        else:
            raise ValueError(f"'emotion' column not found in {self.data_path}")      
        for attr in self.config.attributes:
            if attr not in self.data.columns:
                raise ValueError(f"'{attr}' column not found in {self.data_path}")   

    def __len__(self):
        return len(self.data)


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
        if not self.config.expert_mode == 'sptoken':
            if stage in (None, "fit"):
                self.train_dataset = GEA_Dataset(self.config, self.train_path, self.emotion_or_appraisal, tokenizer= self.tokenizer, max_token_len=self.max_token_length, use_embeddings=self.use_input_embeddings)
                self.val_dataset = GEA_Dataset(self.config, self.val_path, self.emotion_or_appraisal,tokenizer= self.tokenizer,  max_token_len=self.max_token_length, use_embeddings=self.use_input_embeddings)
            if stage == 'predict':
                self.test_dataset = GEA_Dataset(self.config,self.test_path, self.emotion_or_appraisal,tokenizer= self.tokenizer,  max_token_len=self.max_token_length, use_embeddings=self.use_input_embeddings)
        else:
            if stage in (None, "fit"):
                self.train_dataset = SPToken_Dataset(self.config, self.train_path, self.emotion_or_appraisal, tokenizer= self.tokenizer, max_token_len=self.max_token_length, use_embeddings=self.use_input_embeddings)
                self.val_dataset = SPToken_Dataset(self.config, self.val_path, self.emotion_or_appraisal,tokenizer= self.tokenizer,  max_token_len=self.max_token_length, use_embeddings=self.use_input_embeddings)
            if stage == 'predict':
                self.test_dataset = SPToken_Dataset(self.config,self.test_path, self.emotion_or_appraisal,tokenizer= self.tokenizer,  max_token_len=self.max_token_length, use_embeddings=self.use_input_embeddings)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.train_batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.val_batch_size, num_workers=4,  shuffle=False)
        # return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4,persistent_workers=True, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.val_batch_size, num_workers=4, shuffle=False)
