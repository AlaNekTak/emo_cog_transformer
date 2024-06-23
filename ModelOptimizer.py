import os
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from Config import Config, NoOpCallback, CustomCallback, Log
from Data import GEA_Data_Module, GEA_Dataset
from Model import GEA_Emotion_Classifier, MixExp_Emotion_Classifier


""" ModelOptimizer """
class ModelOptimizer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.train_path = config.train_path
        self.val_path = config.val_path
        self.use_input_embeddings = config.use_input_embeddings
        self.epochs = config.epochs
        self.checkpoint_callback = ModelCheckpoint(monitor="val_loss", #validation loss _epoch
                                            filename='model-epoch-{epoch:02d}-val_loss-{val_loss:.2f}',
                                            save_top_k=7,  # Save only the best model based on val_loss
                                            mode='min',# Save the model with the minimum val_loss
                                            auto_insert_metric_name=False
                                            ) 
        self.custom_callback = CustomCallback(logger, checkpoint_callback=self.checkpoint_callback)
        self.early_stopping = EarlyStopping(
            monitor='val_loss', #validation loss _epoch
            patience=3,  # Stopping after 4 epochs of no improvement
            verbose=True,
            mode='min'
        )
        # Start TensorBoard in the background
        self.tb_logger = TensorBoardLogger(save_dir=os.getcwd(), version=datetime.now().strftime("%Y_%m_%d__%H_%M_%S"), name='lightning_logs')
        
    def objective(self, trial):  
        # Configurable hyperparameters
        self.config.lr = trial.suggest_categorical('lr', self.config.optimizer_lr) # trial.suggest_float('lr', 1e-5, 1e-3, log=True) 
        self.config.train_batch_size = trial.suggest_categorical('train_batch_size', self.config.optimizer_batch_size)
        self.logger.info(f"Trial {trial.number}, lr: {self.config.lr}, train batch size: {self.config.train_batch_size}")
        if not self.config.emotion_or_appraisal == 'both':
            model = GEA_Emotion_Classifier(self.config).to(self.config.device)
        else: 
            model = MixExp_Emotion_Classifier(self.config).to(self.config.device)

        # if self.config.load_from_checkpoint:
        #     checkpoint = torch.load(self.config.checkpoint_dir, map_location=self.config.device)
        #     model.load_state_dict(checkpoint['state_dict']) # strict=False
        #     # model.load_from_checkpoint(self.config.checkpoint_dir, map_location=self.config.device)

        trainer = pl.Trainer(max_epochs=self.epochs, 
                            accelerator="auto",
                            # callbacks=[self.checkpoint_callback],  # Add the checkpoint callback here
                            callbacks=[self.custom_callback,self.checkpoint_callback], #  self.early_stopping ,  RichProgressBar() , TQDMProgressBar(refresh_rate=1)
                            # profiler="simple", # to profile standard training events
                            # fast_dev_run=30,#runs 7 predict batches and program ends
                            log_every_n_steps=5, 
                            # num_sanity_val_steps=10, 
                            # strategy=DDPStrategy(find_unused_parameters=True),
                            devices=1,
                            # accumulate_grad_batches=1,
                            logger = self.tb_logger
                            )
        
        GEA_data_module = GEA_Data_Module(self.config)
        GEA_data_module.setup()
        trainer.fit(model, GEA_data_module)

        # val_loss = trainer.callback_metrics["val_loss"].item() # validation loss _epoch
        # self.logger.info(f'val_loss: {val_loss}')
        best_val_loss = self.checkpoint_callback.best_model_score.item()  
        self.logger.info(f'Best validation loss: {best_val_loss}')
        return best_val_loss
