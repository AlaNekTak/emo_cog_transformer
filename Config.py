import torch
import pandas as pd
import os
from datetime import datetime
from transformers import  AutoTokenizer
from peft import  LoraConfig
import logging
from pytorch_lightning.callbacks import  Callback


class Config:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.use_input_embeddings = True
        self.load_from_checkpoint = False
        # self.checkpoint_dir = 'checkpoint_emotion.pth.tar'
        # self.checkpoint_dir = '/home1/nekouvag/local_files/lightning_logs/mixExpert2/checkpoints/model-epoch-16-val_loss-15.62.ckpt'
        # probe case:
        self.checkpoint_dir = '/home1/nekouvag/projects/emo_cog/lightning_logs/2024_06_30__13_02_59/checkpoints/model-epoch-00-val_loss-2.65.ckpt'
        self.inspect_data = False
        self.quick_test = False
        self.mode = args.mode
        self.expert_mode = args.expert_mode
        self.random_state = args.random_state
        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size
        self.train_val_split = args.train_val_split
        self.hparam_trials =  args.hparam_trials
        self.optimizer_lr = args.optimizer_lr
        self.optimizer_batch_size = args.optimizer_batch_size
        self.max_length = args.max_length
        self.lr = args.learning_rate
        self.warmup= args.warmup
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.dropout_rate = args.dropout_rate
        self.log_path = args.log_path
        self.train_path = args.train_data_path
        self.val_path = args.val_data_path
        self.test_path = args.test_data_path
        self.data_encoding = args.data_encoding
        self.gate_mechanism = 'soft'
        self.train_data = pd.read_csv(self.train_path, encoding=self.data_encoding)
        self.val_data = pd.read_csv(self.val_path, encoding=self.data_encoding)
        self.text = 'hidden_emo_text'
        self.peft_config_roberta = LoraConfig(r=4, lora_alpha=16, lora_dropout=0.1, bias="none")
        self.emotion_or_appraisal = args.emotion_or_appraisal
        self.emo_dict = {
            "anger": 0, "boredom": 1, "disgust": 2, "fear": 3, "guilt": 4, "joy": 5,
            "no-emotion": 6, "pride": 7, "relief": 8, "sadness": 9, "shame": 10,
            "surprise": 11, "trust": 12
        }
        if self.emotion_or_appraisal == 'emotion' or self.emotion_or_appraisal == 'both':
            # Map the 'emotion' column in train_data using the emo_dict
            self.train_data['emotion'] = self.train_data['emotion'].map(self.emo_dict).astype(int)
            self.class_num = len(self.emo_dict)
            # print("Emotion dictionary mapping:", self.emo_dict)
            self.val_data['emotion'] = self.val_data['emotion'].map(self.emo_dict).astype(int)
            self.emo_attributes = ['emotion']
            self.n_emo_attributes = len(self.emo_attributes) # if multi-label

        if self.emotion_or_appraisal == 'appraisal' or self.emotion_or_appraisal == 'both':
            # self.attributes = ['predict_event', 'pleasantness', 'attention','other_responsblt']
            self.attributes = ['predict_event', 'pleasantness', 'attention',
            'other_responsblt', 'chance_control', 'social_norms']
            self.n_attributes = len(self.attributes) # if multi-label

        self.train_size = len(self.train_data)

""" Custom callback including checkpointing"""
class CustomCallback(Callback):
    def __init__(self, logger,checkpoint_callback):
        self.checkpoint_callback = checkpoint_callback
        self.logger =logger

    def on_train_start(self, trainer, pl_module):
        self.logger.info("\n\nTraining is starting...\n--------------------------")

    def on_epoch_end(self, trainer, pl_module):
        self.logger.info(f"Epoch {trainer.current_epoch} has ended.\n")
        # Log epoch and validation loss
        if "loss" in trainer.callback_metrics:
            train_loss = trainer.callback_metrics["loss"]
            self.logger.info(f"Training loss: {train_loss}\n")
        if "val_loss" in trainer.callback_metrics:
            val_loss = trainer.callback_metrics["val_loss"]
            self.logger.info(f"Validation loss: {val_loss}\n")
    
    def on_validation_end(self, trainer, pl_module):
        # Log validation metrics if available
        if "val_loss" in trainer.callback_metrics:
            val_loss = trainer.callback_metrics["val_loss"]
            self.logger.info(f'Validation ended. Epoch {trainer.current_epoch}  val_loss: {val_loss}\n')
        else:
            self.logger.info('Validation ended. No val_loss available.\n')

    def on_train_end(self, trainer, pl_module):
        self.logger.info("Training has finished.\n")


class NoOpCallback(Callback):
    def on_validation_end(self, *args, **kwargs):
        pass
    

class Log:
    def __init__(self, config):
        self.log_path = os.path.join(config.log_path,
                                      f'mode-{config.mode}_type-{config.emotion_or_appraisal}_date-{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}.txt')
        self.logger = self._setup_logging()
        self._log_config(config)  # Call to log config upon initialization

    def _setup_logging(self):

        if os.path.exists(self.log_path):
            os.remove(self.log_path)

        # Configure the root logger
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            handlers=[
                                logging.FileHandler(self.log_path),
                                logging.StreamHandler()
                            ])
        
        # Return the root logger
        logger = logging.getLogger()

        return logger

    def _log_config(self, config):
        # Log all attributes in the config object
        for attr_name, attr_value in vars(config).items():
            self.logger.info(f'CONFIG {attr_name}: {attr_value}\n')


    def close_logging(self, logger):
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)
