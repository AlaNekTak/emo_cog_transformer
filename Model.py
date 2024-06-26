import math
import torch
import torch.nn as nn
from transformers import AutoModel,get_cosine_schedule_with_warmup
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW # instead of from transformers


""" Model"""
class GEA_Emotion_Classifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pretrained_model = AutoModel.from_pretrained(config.model_name, return_dict = True)
        # self.pretrained_model = get_peft_model(AutoModel.from_pretrained(self.config.model_name,return_dict = True),
        #                                         self.config.peft_config_roberta)
        self.hidden = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        if config.emotion_or_appraisal =='emotion':
            self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.config.class_num)
            self.loss_func = nn.CrossEntropyLoss() # averaging loss over the batch, batch size indep results
            # self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')
            # self.loss_func = nn.NLLLoss()
        elif config.emotion_or_appraisal == 'appraisal':
            self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.config.n_attributes)
            self.loss_func =  torch.nn.MSELoss()
        
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.use_input_embeddings = self.config.use_input_embeddings
        self.save_hyperparameters()
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None):
        if self.use_input_embeddings:
            output = self.pretrained_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = torch.mean(output.last_hidden_state, 1)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.hidden(pooled_output)
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # log_probs = F.log_softmax(logits, dim=1)  # Apply softmax to convert to log probabilities
        # calculate loss
        loss = 0
        if labels is not None:
            # loss = self.loss_func(log_probs, labels)  # Use log probabilities here
            if self.config.emotion_or_appraisal == 'emotion':
                labels = labels.view(-1)
                assert labels.dim() == 1, "Labels shape mismatch: labels should be squeezed to 1 dimension"
                assert logits.shape == (labels.shape[0],self.config.class_num), "Mismatch in batch size between logits and labels"
                loss = self.loss_func(logits, labels) # torch.Size([batch, 1]) to torch.Size([batch])
            elif self.config.emotion_or_appraisal == 'appraisal':
                loss = self.loss_func(logits.view(-1, self.config.n_attributes), labels.view(-1, self.config.n_attributes))        
        return loss, logits #log_probs 
    
    def training_step(self, batch, batch_index):
        loss, outputs = self(**batch)
        self.log("train loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # writer.add_scalar("Loss/train", loss)
        return {"loss":loss, "predictions":outputs, "labels": batch["labels"]}

    def validation_step(self, batch, batch_index):
        loss, outputs= self(**batch)
        self.log("val_loss", loss, prog_bar = True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # writer.add_scalar("Loss/val", loss)
        return {"val_loss": loss, "predictions":outputs, "labels": batch["labels"]}

    def predict_step(self, batch, batch_index):
        _, outputs = self(**batch)
        return {"predictions":outputs, "labels": batch["labels"]}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        total_steps = self.config.train_size/self.config.train_batch_size
        warmup_steps = math.floor(total_steps * self.config.warmup)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer],[scheduler]

    def extract_embeddings(self, input_ids=None, attention_mask=None):
        self.eval()
        with torch.no_grad():
            output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = output.last_hidden_state
        return embeddings
    

class MixExp_Emotion_Classifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hard_gate_num = 3
        self.use_input_embeddings = self.config.use_input_embeddings
        
        self.pretrained_model = AutoModel.from_pretrained(self.config.model_name, return_dict=True)

        # Individual appraisal hidden layers
        self.appraisal_hidden = nn.ModuleList([nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size) for _ in range(self.config.n_attributes)])
        self.appraisal_classifiers = nn.ModuleList([nn.Linear(self.pretrained_model.config.hidden_size, 1) for _ in range(self.config.n_attributes)])
        
        if self.config.gate_mechanism == 'hard':
            # Selection layer that decides which appraisal outputs to use for emotion classification
            self.selection_weights = nn.Parameter(torch.randn(self.config.n_attributes, self.hard_gate_num))  # Random initialization
            self.emotion_hidden = nn.Linear(self.hard_gate_num * self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        
        elif self.config.gate_mechanism == 'soft':
            # # static
            # self.gate_weights = nn.Parameter(torch.randn(self.config.n_attributes, 1))  # Initialize gate weights

            # dynamic
            self.dynamic_gate = DynamicSoftGate(self.pretrained_model.config.hidden_size, config.n_attributes)
            self.emotion_hidden = nn.Linear(self.config.n_attributes * self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size) # Emotion classification components
        
        self.emotion_classifier = nn.Linear(self.pretrained_model.config.hidden_size, self.config.class_num)

        # Loss functions
        self.appraisal_loss_func = nn.MSELoss()
        self.emotion_loss_func = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(self.config.dropout_rate)
        
        ## Weight initialization
        self.save_hyperparameters()
        self.initialize_weights()

    def forward(self, input_ids=None, attention_mask=None, appraisal_labels=None, emotion_labels=None, inputs_embeds=None):
        if self.use_input_embeddings:
            outputs = self.pretrained_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = torch.mean(outputs.last_hidden_state, 1)

        appraisal_outputs = [F.relu(hidden(pooled_output)) for hidden in self.appraisal_hidden]
        appraisal_logits = [self.dropout(classifier(a_out)) for classifier, a_out in zip(self.appraisal_classifiers, appraisal_outputs)]
        
        if self.config.gate_mechanism == 'hard':
            # Apply softmax to selection weights and select appraisal features
            selection_weights = F.softmax(self.selection_weights, dim=0)
            emotion_input = torch.matmul(selection_weights.T, torch.stack(appraisal_outputs)) #selected features
        
        elif self.config.gate_mechanism == 'soft':
            # # Apply soft gating to each appraisal output # sigmoid to have independent binary gates or (prob to 1)
            # gated_appraisal_outputs = [weight * output for weight, output in zip(torch.sigmoid(self.gate_weights), appraisal_outputs)] 
            
            # # Static: Apply softmax and ensure it sums to 1 to have relative gates
            # gate_weights_stat = F.softmax(self.gate_weights.squeeze(), dim=0)  
            # gated_appraisal_outputs_stat = [weight * output for weight, output in zip(gate_weights_stat, appraisal_outputs)]
            # emotion_input_stat = torch.cat(gated_appraisal_outputs_stat, dim=1)

            # Dynamic
            gate_weights = self.dynamic_gate(pooled_output)  # Shape: [batch_size, n_attributes]
            gated_appraisal_outputs = [weight.unsqueeze(1) * output for weight, output in zip(gate_weights.t(), appraisal_outputs)]
            emotion_input = torch.cat(gated_appraisal_outputs, dim=1)

        emotion_pooled = self.dropout(F.relu(self.emotion_hidden(emotion_input)))
        emotion_logits = self.emotion_classifier(emotion_pooled)

        total_loss = 0
        loss_dict = {}
        if appraisal_labels is not None and emotion_labels is not None:
            emotion_labels = emotion_labels.view(-1)
            assert emotion_logits.shape == (emotion_labels.shape[0],self.config.class_num), "Mismatch in batch size between logits and labels"
            emotion_loss = self.emotion_loss_func(emotion_logits, emotion_labels)  # Assuming emotion labels are the last
            appraisal_losses = [self.appraisal_loss_func(logits.view(-1), appraisal_labels[:, i]) for i, logits in enumerate(appraisal_logits)] # or logits.view(-1,1), appraisal_labels[:, i:i+1] [10, 1], but [:, i] would make it [10]
            total_loss += emotion_loss 
            total_loss += sum(appraisal_losses)
            loss_dict['emotion'] = emotion_loss.item()
            loss_dict['appraisals'] = [loss.item() for loss in appraisal_losses]

        return total_loss, emotion_logits, appraisal_logits, gate_weights, loss_dict

    def training_step(self, batch, batch_index):
        total_loss, emotion_logits, appraisal_logits,gate_weights, loss_dict = self(**batch)
        self.log("train_loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("emotion_loss", loss_dict['emotion'], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        for idx, loss in enumerate(loss_dict['appraisals']):
                self.log(f'appraisal_{idx}_loss', loss, on_step=False, on_epoch=True)

        if self.config.gate_mechanism == 'soft':
            mean_gate_weights = gate_weights.mean(dim=0)
            for idx, weight in enumerate(mean_gate_weights):
                self.log(f'gate_weight_{idx}', weight, on_step=False, on_epoch=True)

        return {"loss": total_loss, 
                "emotion_logits":emotion_logits,
                "appraisal_logits":appraisal_logits,
                "emotion_labels": batch["emotion_labels"], 
                "appraisal_labels": batch["appraisal_labels"],                
                "gate_weights": gate_weights
                }
    
    def validation_step(self, batch, batch_index):
        total_loss, emotion_logits, appraisal_logits, gate_weights, loss_dict= self(**batch)
        self.log("val_loss", total_loss, prog_bar = True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return {"val_loss": total_loss, "emotion_logits":emotion_logits,"appraisal_logits":appraisal_logits,
                 "emotion_labels": batch["emotion_labels"], "appraisal_labels": batch["appraisal_labels"]}

    def predict_step(self, batch, batch_index):
        _, emotion_logits, appraisal_logits, gate_weights, _ = self(**batch)
        return {
                "emotion_logits":emotion_logits,
                "appraisal_logits":appraisal_logits,
                "emotion_labels": batch["emotion_labels"], 
                "appraisal_labels": batch["appraisal_labels"],
                "gate_weights": gate_weights
                }

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        total_steps = self.config.train_size/self.config.train_batch_size
        warmup_steps = math.floor(total_steps * self.config.warmup)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer],[scheduler]

    def initialize_weights(self):
        for hidden in self.appraisal_hidden:
            nn.init.xavier_uniform_(hidden.weight)
        for classifier in self.appraisal_classifiers:
            nn.init.xavier_uniform_(classifier.weight)
        nn.init.xavier_uniform_(self.emotion_hidden.weight)
        nn.init.xavier_uniform_(self.emotion_classifier.weight)

    def on_train_epoch_end_static(self):
        # Log the gate weights at the end of each epoch
        if self.config.gate_mechanism == 'hard':
            weights = F.softmax(self.selection_weights, dim=0).data
        elif self.config.gate_mechanism == 'soft':
            # weights = torch.sigmoid(self.gate_weights).data
            weights = F.softmax(self.gate_weights.squeeze(), dim=0)

        # Use PyTorch Lightning's logger or any other logging mechanism
        for idx, weight in enumerate(weights.squeeze()):
            print(f'gate_weight{idx}-{self.config.attributes[idx]}', weight.item())
            self.log(f'gate_weight_{idx}-{self.config.attributes[idx]}', weight.mean().item(), on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self):
        if self.config.gate_mechanism == 'soft':
            for idx in range(self.config.n_attributes):
                # Retrieve each gate weight average logged during the epoch
                print(f'{self.config.attributes[idx]} weight: ',self.trainer.callback_metrics.get(f'gate_weight_{idx}').cpu().numpy())
                print(f'{self.config.attributes[idx]} loss: ',self.trainer.callback_metrics.get(f'appraisal_{idx}_loss').cpu().numpy())
            print('emotion loss: ',self.trainer.callback_metrics.get(f'emotion_loss').cpu().numpy())


class DynamicSoftGate(nn.Module):
    def __init__(self, input_size, num_attributes):
        super(DynamicSoftGate, self).__init__()
        # Initialize a linear layer that maps the input features to gate weights
        self.gate_layer = nn.Linear(input_size, num_attributes)

    def forward(self, x):
        # Apply sigmoid to ensure the output weights are between 0 and 1
        # gate_weights = torch.sigmoid(self.gate_layer(x))
        
        # Use softmax to normalize weights to sum to 1
        gate_weights = F.softmax(self.gate_layer(x), dim=1)
        return gate_weights



class DoubleExp_Emotion_Classifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_input_embeddings = self.config.use_input_embeddings
        
        self.pretrained_model = AutoModel.from_pretrained(self.config.model_name, return_dict=True)

        # Appraisal path
        self.appraisal_hidden = nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        self.appraisal_classifier = nn.Linear(self.pretrained_model.config.hidden_size, 1)

        self.emotion_hidden = nn.Linear(2 * self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size) # Emotion classification components
        
        self.emotion_classifier = nn.Linear(self.pretrained_model.config.hidden_size, self.config.class_num)

        self.dynamic_gate = DynamicSoftGate(self.pretrained_model.config.hidden_size, 2)
        # Loss functions
        self.appraisal_loss_func = nn.MSELoss()
        self.emotion_loss_func = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(self.config.dropout_rate)
        
        ## Weight initialization
        self.save_hyperparameters()
        self.initialize_weights()

    def forward(self, input_ids=None, attention_mask=None, appraisal_labels=None, emotion_labels=None, inputs_embeds=None):
        if self.use_input_embeddings:
            outputs = self.pretrained_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)

        mean_last_hidden = torch.mean(outputs.last_hidden_state, 1)

        # Appraisal processing
        appraisal_output = F.relu(self.appraisal_hidden(mean_last_hidden))
        appraisal_logits = self.dropout(self.appraisal_classifier(appraisal_output))
        
        # Dynamic
        gate_weights = self.dynamic_gate(mean_last_hidden)  # Shape: [batch_size, n_attributes]
        concat_layer = torch.cat([
                                    gate_weights[:, 0].unsqueeze(1) * appraisal_output,
                                    gate_weights[:, 1].unsqueeze(1) * mean_last_hidden
                                ], dim=1)

        emotion_pooled = self.dropout(F.relu(self.emotion_hidden(concat_layer)))
        emotion_logits = self.emotion_classifier(emotion_pooled)

        total_loss = 0
        loss_dict = {}

        if appraisal_labels is not None and emotion_labels is not None:
            emotion_labels = emotion_labels.view(-1)
            assert emotion_logits.shape == (emotion_labels.shape[0],self.config.class_num), "Mismatch in batch size between logits and labels"
            emotion_loss = self.emotion_loss_func(emotion_logits, emotion_labels)  # Assuming emotion labels are the last
            appraisal_loss = self.appraisal_loss_func(appraisal_logits.view(-1), appraisal_labels)
            total_loss += emotion_loss + appraisal_loss
            loss_dict['emotion'] = emotion_loss.item()
            loss_dict['appraisals'] = appraisal_loss.item()

        return total_loss, emotion_logits, appraisal_logits, gate_weights, loss_dict

    def training_step(self, batch, batch_index):
        total_loss, emotion_logits, appraisal_logits,gate_weights, loss_dict = self(**batch)
        self.log("train_loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("emotion_loss", loss_dict['emotion'], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        for idx, loss in enumerate(loss_dict['appraisals']):
                self.log(f'appraisal_{idx}_loss', loss, on_step=False, on_epoch=True)

        if self.config.gate_mechanism == 'soft':
            mean_gate_weights = gate_weights.mean(dim=0)
            for idx, weight in enumerate(mean_gate_weights):
                self.log(f'gate_weight_{idx}', weight, on_step=False, on_epoch=True)

        return {"loss": total_loss, 
                "emotion_logits":emotion_logits,
                "appraisal_logits":appraisal_logits,
                "emotion_labels": batch["emotion_labels"], 
                "appraisal_labels": batch["appraisal_labels"],                
                "gate_weights": gate_weights
                }
    
    def validation_step(self, batch, batch_index):
        total_loss, emotion_logits, appraisal_logits, gate_weights, loss_dict= self(**batch)
        self.log("val_loss", total_loss, prog_bar = True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return {"val_loss": total_loss, "emotion_logits":emotion_logits,"appraisal_logits":appraisal_logits,
                 "emotion_labels": batch["emotion_labels"], "appraisal_labels": batch["appraisal_labels"]}

    def predict_step(self, batch, batch_index):
        _, emotion_logits, appraisal_logits, gate_weights, _ = self(**batch)
        return {
                "emotion_logits":emotion_logits,
                "appraisal_logits":appraisal_logits,
                "emotion_labels": batch["emotion_labels"], 
                "appraisal_labels": batch["appraisal_labels"],
                "gate_weights": gate_weights
                }

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        total_steps = self.config.train_size/self.config.train_batch_size
        warmup_steps = math.floor(total_steps * self.config.warmup)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer],[scheduler]

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.appraisal_hidden.weight)
        nn.init.xavier_uniform_(self.appraisal_classifier.weight)
        nn.init.xavier_uniform_(self.emotion_hidden.weight)
        nn.init.xavier_uniform_(self.emotion_classifier.weight)

    def on_train_epoch_end_static(self):
        # Log the gate weights at the end of each epoch
        if self.config.gate_mechanism == 'hard':
            weights = F.softmax(self.selection_weights, dim=0).data
        elif self.config.gate_mechanism == 'soft':
            # weights = torch.sigmoid(self.gate_weights).data
            weights = F.softmax(self.gate_weights.squeeze(), dim=0)

        # Use PyTorch Lightning's logger or any other logging mechanism
        for idx, weight in enumerate(weights.squeeze()):
            print(f'gate_weight{idx}-{self.config.attributes[idx]}', weight.item())
            self.log(f'gate_weight_{idx}-{self.config.attributes[idx]}', weight.mean().item(), on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self):
        if self.config.gate_mechanism == 'soft':
            for idx in range(self.config.n_attributes):
                # Retrieve each gate weight average logged during the epoch
                print(f'{self.config.attributes[idx]} weight: ',self.trainer.callback_metrics.get(f'gate_weight_{idx}').cpu().numpy())
                print(f'{self.config.attributes[idx]} loss: ',self.trainer.callback_metrics.get(f'appraisal_{idx}_loss').cpu().numpy())
            print('emotion loss: ',self.trainer.callback_metrics.get(f'emotion_loss').cpu().numpy())

    # def on_validation_epoch_end(self):
    #     if self.config.gate_mechanism == 'soft':
    #         for idx in range(self.config.n_attributes):
    #             # Retrieve each gate weight average logged during the epoch
    #             print(f'{self.config.attributes[idx]} weight: ',self.trainer.callback_metrics.get(f'gate_weight_{idx}').cpu().numpy())
    #             print(f'{self.config.attributes[idx]} loss: ',self.trainer.callback_metrics.get(f'appraisal_{idx}_loss').cpu().numpy())
    #         print('emotion loss: ',self.trainer.callback_metrics.get(f'emotion_loss').cpu().numpy())