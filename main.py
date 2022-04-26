import os

from data_preprocessing import reduce_dataset, split_data, preprocess_data, DATA_BIN_DIR, Dataset, get_tokenized_data
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset
from models import get_model_list
from torch.utils.data import DataLoader
import torch
import pandas as pd

FAIRSEQ_MODEL_PATH = './models'
CHECKPOINT_DIR = './checkpoint'

SAVED_DATA_PATH = './data/'
Train_csv = 'train_data.csv'
Val_csv = 'val_data.csv'

def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)

    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator


if __name__ == '__main__':
    batch_size = 4
    learning_rate = 0.005
    epochs = 0
    max_update = 10  # stop training at specified update

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # data encoding
    file_path, raw_data, eliminated_data = reduce_dataset()
    train_df, val_df = split_data(raw_data)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    train_dataset, test_dataset, data_collator = load_dataset(SAVED_DATA_PATH+Train_csv,SAVED_DATA_PATH+Val_csv, tokenizer)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = AutoModelForCausalLM.from_pretrained("KoboldAI/fairseq-dense-125M")
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset3z
    )

    trainer.train()

    '''
    
    imdb = dataset = load_dataset('csv', data_files={'train': [SAVED_DATA_PATH + Train_csv], 'test': SAVED_DATA_PATH + Val_csv})
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForCausalLM.from_pretrained("KoboldAI/fairseq-dense-125M")
     
     
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    '''
