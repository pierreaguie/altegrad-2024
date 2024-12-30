import os
from tqdm import tqdm
import random
import re

import torch
from transformers import BertTokenizer, BertModel

import warnings


random.seed(32)


def extract_numbers(text):
    # Use regular expression to find integers and floats
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    # Convert the extracted numbers to float
    return [float(num) for num in numbers]


def extract_feats(file):
    stats = []
    fread = open(file,"r")
    line = fread.read()
    line = line.strip()
    stats = extract_numbers(line)
    fread.close()
    return stats


def extract_BERT(text):
    device = "cpu"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(device)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
    
    cond = outputs.last_hidden_state[:, 0, :]
    return cond