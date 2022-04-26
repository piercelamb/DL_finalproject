import os

from data_preprocessing import reduce_dataset, split_data, preprocess_data, DATA_BIN_DIR, Dataset, get_tokenized_data
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import Dataset, load_dataset
from models import get_model_list
from torch.utils.data import DataLoader
import torch
import pandas as pd
from tqdm.notebook import tqdm

FAIRSEQ_MODEL_PATH = './models'
CHECKPOINT_DIR = './checkpoint'

SAVED_DATA_PATH = './data/'
Train_csv = 'train_data.csv'
Val_csv = 'val_data.csv'


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        a = self.labels.itmes()
        item['label'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def train(dataloader, model, optimizer, num_train_epochs, data_len):
    global_step = 0
    #t_total = len(dataloader) * num_train_epochs  # total number of training steps

    # put the model in training mode
    model.train()
    for epoch in range(num_train_epochs):
        print("Epoch:", epoch)
        running_loss = 0.0
        correct = 0
        for batch in tqdm(dataloader): #TODO get tqdm working right
            # forward pass
            outputs = model(**batch)
            loss = outputs.loss

            running_loss += loss.item()
            predictions = outputs.logits.argmax(-1)
            correct += (predictions == batch['labels']).float().sum()

            # backward pass to get the gradients
            loss.backward()

            # update
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        print("Loss:", running_loss / batch["input_ids"].shape[0])
        accuracy = 100 * correct / data_len
        print("Training accuracy:", accuracy.item())

if __name__ == '__main__':
    batch_size = 16
    learning_rate = 0.005
    epochs = 2
    max_update = 10  # stop training at specified update

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # data encoding
    file_path, raw_data, eliminated_data = reduce_dataset()
    train_df, val_df = split_data(raw_data)

    train_tokenized, val_tokenized = get_tokenized_data(train_df, val_df)
    train_dataloader = torch.utils.data.DataLoader(train_tokenized, batch_size=batch_size)

    model = AutoModelForCausalLM.from_pretrained("KoboldAI/fairseq-dense-125M")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train(train_dataloader, model, optimizer, epochs, len(train_df))

    # train_texts = train_texts.astype(str).values.tolist()
    # val_texts = val_texts.astype(str).values.tolist()
    # train_labels = train_labels.astype(str).values.tolist()
    # val_labels = val_labels.astype(str).values.tolist()
    # eliminated_data_text = eliminated_data['text'].astype(str).values.tolist()
    # eliminated_data_label = eliminated_data['label'].astype(str).values.tolist()
    #
    # training_data = []
    # validation_data = []
    # testing__data = []
    #
    # size_training = len(train_texts)
    # for i in range(0, size_training):
    #     string = 'Question: ' + train_texts[i] + ' Answer: ' + train_labels[i]
    #     training_data.append(string)
    #
    # size_validation = len(train_labels)
    # for i in range(0, size_validation):
    #     string = 'Question: ' + val_texts[i] + ' Answer: ' + val_labels[i]
    #     validation_data.append(string)
    #
    # size_eliminated = len(eliminated_data_text)
    # for i in range(0, size_eliminated):
    #     string = 'Question: ' + eliminated_data_text[i] + ' Answer: ' + eliminated_data_label[i]
    #     testing__data.append(string)
    #
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #
    #
    #
    # print("1")
    # train_encodings = tokenizer(training_data, max_length=50, padding='max_length', truncation=True,
    #                             return_tensors='pt')
    # print("2")
    # val_encodings = tokenizer(validation_data, max_length=50, padding='max_length', truncation=True, return_tensors='pt')
    # print("3")
    # val_label_encodings = tokenizer(testing__data, max_length=50, padding='max_length', truncation=True,
    #                                 return_tensors='pt')
    #
    # # TODO: Convert the encodings to a dataset....
    # '''
    # train_dataset = IMDbDataset(train_encodings, train_label_encoding)
    # val_dataset = IMDbDataset(val_encodings, val_label_encodings)
    # test_dataset = IMDbDataset(test_encodings, test_label_encoding)'''
    #
    # from torch.utils.data import DataLoader
    # from transformers import DistilBertForSequenceClassification, AdamW
    #
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #
    # model = AutoModelForCausalLM.from_pretrained("KoboldAI/fairseq-dense-125M")
    # model.to(device)
    # model.train()
    #
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    #
    # training_args = TrainingArguments(
    #     output_dir='./results',  # output directory
    #     num_train_epochs=3,  # total number of training epochs
    #     per_device_train_batch_size=16,  # batch size per device during training
    #     per_device_eval_batch_size=64,  # batch size for evaluation
    #     warmup_steps=500,  # number of warmup steps for learning rate scheduler
    #     weight_decay=0.01,  # strength of weight decay
    #     logging_dir='./logs',  # directory for storing logs
    #     logging_steps=10,
    # )
    # trainer = Trainer(
    #     model=model,  # the instantiated 🤗 Transformers model to be trained
    #     args=training_args,  # training arguments, defined above
    #     train_dataset=train_dataset,  # training dataset
    #     eval_dataset=val_dataset  # evaluation dataset3z
    # )
    #
    # trainer.train()
    #
    # '''
    #
    # imdb = dataset = load_dataset('csv', data_files={'train': [SAVED_DATA_PATH + Train_csv], 'test': SAVED_DATA_PATH + Val_csv})
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # tokenized_imdb = imdb.map(preprocess_function, batched=True)
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # model = AutoModelForCausalLM.from_pretrained("KoboldAI/fairseq-dense-125M")
    #
    #
    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=16,
    #     num_train_epochs=5,
    #     weight_decay=0.01,
    # )
    #
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_imdb["train"],
    #     eval_dataset=tokenized_imdb["test"],
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    # )
    #
    # trainer.train()
    #
    # '''
