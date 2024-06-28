import csv
import torch
import torch.nn as nn
import pandas as pd
import os, sys
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from transformers import AutoModel,get_cosine_schedule_with_warmup, Trainer, TrainingArguments, AutoTokenizer, TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType, peft_model
from torch.utils.data import Dataset
import logging
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, r2_score
from torchmetrics.functional.classification import auroc
import torch.nn.functional as F
from sklearn import metrics
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW # instead of from transformers
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback, TQDMProgressBar, RichProgressBar
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
import optuna
import argparse
from scipy.stats import mode


from Config import Config, NoOpCallback, CustomCallback, Log
from Data import GEA_Data_Module, GEA_Dataset
from Model import GEA_Emotion_Classifier, MixExp_Emotion_Classifier
from ModelOptimizer import ModelOptimizer
from TrainTest import test, train, split_train_val_test, quick_test, inspect_data


# import IPython
# import subprocess
# import torch.distributed as dist
# from joblib import Parallel, delayed
# import multiprocessing

def parse_args(): 
    parser = argparse.ArgumentParser(description="Emotion and Appraisal Prediction Model Training")
    # Adding arguments
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length - 512")
    parser.add_argument("--hparam_trials", type=int, default=1, help="Number of hyperparameter trials")
    parser.add_argument("--optimizer_lr", type=float, default=[1e-5], help="Learning rates given to optimizer")
    parser.add_argument("--optimizer_batch_size", type=int, default=[32], help="Training batch sizes given to the optimizer")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=32, help="Validation batch size")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--warmup", type=float, default=0.1, help="Warmup")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="weight decay")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs for training")
    parser.add_argument("--data_encoding", type=str, default='ISO-8859-1', help="Data encoding type")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for train-val division")
    parser.add_argument("--model_arch", type=str, default='Hierarchical_Emotion_Emotion', choices=['Emotion_Only', 'Pretrained_Appraisal_post_Emotion', 'Hierarchical_Appraisal_Emotion', 'Text_And_Appraisal_Input', 'Hierarchical_Emotion_Emotion'], help="Type of training to perform")
    parser.add_argument("--forced_appraisal_training", type=str, default='False', choices=['True', 'False'], help="Forced appraisal training mode")
    parser.add_argument("--embedding_usage_mode", type=str, default="last", choices=["average", "last", "first"], help="Embedding usage mode")
    parser.add_argument("--model_name", type=str, default='roberta-base', choices=['distilroberta-base','roberta-base', 'roberta-large', 'mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-2-7b-hf'], help="Model name")
    parser.add_argument("--emotion_or_appraisal", type=str, default='both',choices=['emotion', 'appraislal', 'both'], help='Is it emotion classification or appraisal prediction?')
    parser.add_argument("--train_val_split", type=float, default=0.1, help="val/train split ratio")
    parser.add_argument("--mode", type=str, default='both',choices=['both', 'train_only', 'test_only'], help='Are you training, testing, or both?')
    parser.add_argument("--expert_mode", type=str, default='probe',choices=['double', 'mixed', 'probe'], help='what model arch are you using?')
    # Paths
    parser.add_argument("--train_data_path", type=str, default='data/enVent_new_Data_train.csv', help="Path to the training data")
    parser.add_argument("--val_data_path", type=str, default='data/enVent_new_Data_val.csv', help="Path to the test data")
    parser.add_argument("--test_data_path", type=str, default='data/enVent_new_Data_test.csv', help="Path to the test data")
    parser.add_argument("--valid_result_path", type=str, default='output/validation_results_class.csv', help="Path for validation results (classification)")
    parser.add_argument("--valid_reg_result_path", type=str, default='output/validation_results_regress.csv', help="Path for validation results (regression)")
    parser.add_argument("--log_path", type=str, default='lightning_logs', help="Path to folder containing the training log")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = Config(args)
    log = Log(config)
    logger = log.logger
    logger.info(f"Current Working Directory:{os.getcwd()}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"Using device: {config.device}")
    # dist.init_process_group(backend='nccl')
    seed_everything(args.random_state, workers=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logger.info("TOKENIZERS_PARALLELISM set to false")   

    """ inspect data """
    if config.inspect_data:
       inspect_data(config, logger)

    """ Quick test"""
    if config.quick_test:
        quick_test(config, logger)

    # split_train_val_test(config, 'data/enVent_gen_Data.csv', 'data/enVent_val_Data.csv')
    
    if config.mode == 'train_only':
        best_model_path = train(config, logger)

    elif config.mode == 'test_only':
        test(config.checkpoint_dir,config, logger)

    elif config.mode == 'both':
        best_model_path = train(config, logger)
        # best_model_path = "/home1/nekouvag/local_files/lightning_logs/2024_05_15__09_28_26/checkpoints/model-epoch-00-val_loss-11.64.ckpt"
        test(best_model_path,config, logger)
    else:
        logger.info('Error selecting the training mode! select between train, test, or both!')

    log.close_logging(logger)


