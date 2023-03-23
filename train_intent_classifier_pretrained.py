import pandas as pd
import numpy as np
import json, re
from tqdm import tqdm
from uuid import uuid4

# Torch Modules
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# PyTorch Transformer
from pytorch_transformers import RobertaModel, RobertaTokenizer
from pytorch_transformers import RobertaForSequenceClassification, RobertaConfig

# Load the dataset from a CSV file
raw_dataset = pd.read_csv('data/drink_order_issues.csv')
# dataset = pd.DataFrame(columns = ['utterance', 'label'])
# for i in range(len(raw_dataset['text'])):
#     dataset = dataset.append({'utterance': raw_dataset['intent'][i], 'label': raw_dataset['text'][i]}, ignore_index=True)
dataset_list = []
for i in range(len(raw_dataset['text'])):
    dataset_list.append({'utterance': raw_dataset['text'][i], 'label': raw_dataset['intent'][i]})
dataset = pd.concat([pd.DataFrame(dataset_list[i], index=[i]) for i in range(len(dataset_list))], ignore_index=True)

# convert label string to numerical values
label_to_ix = {'drink spillage': 0,
                'wrong items': 1,
                'quality issues': 2,
                'plain statement': 3}
# for label in dataset.label:
#     for word in label.split():
#         if word not in label_to_ix:
#             label_to_ix[word]=len(label_to_ix)
# print(label_to_ix)

# Loading RoBERTa classes
config = RobertaConfig.from_pretrained('roberta-base')
config.num_labels = len(raw_dataset['intent'].unique())
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification(config)

# Feature Preparation
def prepare_features(seq_1, max_seq_length = 300, 
             zero_pad = False, include_CLS_token = True, include_SEP_token = True):
    ## Tokenzine Input
    tokens_a = tokenizer.tokenize(seq_1)

    ## Truncate
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    ## Initialize Tokens
    tokens = []
    if include_CLS_token:
        tokens.append(tokenizer.cls_token)
    ## Add Tokens and separators
    for token in tokens_a:
        tokens.append(token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    ## Input Mask 
    input_mask = [1] * len(input_ids)
    ## Zero-pad sequence lenght
    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
    return torch.tensor(input_ids).unsqueeze(0), input_mask

# Dataset Loader Classes
class Intents(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe
        
    def __getitem__(self, index):
        utterance = self.data.utterance[index]
        label = self.data.label[index]
        X, _  = prepare_features(utterance)
        y = label_to_ix[self.data.label[index]]
        return X, y
    
    def __len__(self):
        return self.len

train_size = 0.9
train_dataset=dataset.sample(frac=train_size,random_state=200).reset_index(drop=True)
test_dataset=dataset.drop(train_dataset.index).reset_index(drop=True)
print("FULL Dataset: {}".format(dataset.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = Intents(train_dataset)
testing_set = Intents(test_dataset)

# Training Params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.cuda()
# Parameters
params = {'batch_size': 1,
          'shuffle': True,
          'drop_last': False,
          'num_workers': 1}

training_loader = DataLoader(training_set, **params)
testing_loader = DataLoader(testing_set, **params)
loss_function = nn.CrossEntropyLoss()
learning_rate = 1e-05
optimizer = optim.Adam(params =  model.parameters(), lr=learning_rate)

max_epochs = 50
model = model.train()
for epoch in tqdm(range(max_epochs)):
    print("EPOCH -- {}".format(epoch))
    for i, (sent, label) in enumerate(training_loader):
        optimizer.zero_grad()
        sent = sent.squeeze(0)
        if torch.cuda.is_available():
          sent = sent.cuda()
          label = label.cuda()
        output = model.forward(sent)[0]
        _, predicted = torch.max(output, 1)
        
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        
        if i%100 == 0:
            correct = 0
            total = 0
            for sent, label in testing_loader:
                sent = sent.squeeze(0)
                if torch.cuda.is_available():
                  sent = sent.cuda()
                  label = label.cuda()
                output = model.forward(sent)[0]
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted.cpu() == label.cpu()).sum()
            accuracy = 100.00 * correct.numpy() / total
            print('Iteration: {}. Loss: {}. Accuracy: {}%'.format(i, loss.item(), accuracy))

torch.save(model.state_dict(), 'testing_model.pth')
