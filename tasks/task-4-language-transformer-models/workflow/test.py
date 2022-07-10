import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import os
import json

from models.transformer import SentClf
from data.helpers import get_data_loaders, load_dataset
from data.dataset import TamilDataset

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

# Helper function to get predictions from a Data Loader Pytorch object
def model_eval(data, model, args, store_preds=False):
    with torch.no_grad():
        preds, prob_preds, tgts = [], [], []
        for batch in data:
            txt, segment, mask, tgt = batch[0], batch[1], batch[2], batch[3]
            txt, mask, segment = txt.to(device), mask.to(device), segment.to(device)
            tgt = tgt.to(device)

            out = model(txt, mask, segment)

            probs = torch.nn.functional.softmax(out, dim=1)
            pred_idxs = probs.argmax(dim=1).cpu().detach().numpy()
            pred = [1 if p[1] > args.class_threshold else 0 for i,p in enumerate(probs)]
            tgt = tgt.cpu().detach().numpy()
            
            preds.append(pred)
            tgts.append(tgt)
            prob_preds.extend([probs[i].cpu().detach().numpy()[idx] for i,idx in enumerate(pred_idxs)])

    metrics = {}
    tgts = [l for sl in tgts for l in sl]
    preds = [l for sl in preds for l in sl]
    report = classification_report(tgts, preds, output_dict=True)

    return report

# Helper function to load batches of data

def collate_fn(batch):
    bsz = len(batch)
    text_tensor = segment_tensor = mask_tensor = None
    lens = [row['input_ids'].shape[1] for row in batch]
    max_seq_len = max(lens)

    tgt_tensor = torch.cat([row['label'] for row in batch]).long()

    mask_tensor = torch.zeros(bsz, max_seq_len).long()
    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row['input_ids'], input_row['token_type_ids']
        text_tensor[i_batch, :length] = tokens
        segment_tensor[i_batch, :length] = segment
        mask_tensor[i_batch, :length] = 1
    
    return text_tensor, segment_tensor, mask_tensor, tgt_tensor

# Load model

args = torch.load("model_artifacts/args.pt")
model = SentClf(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("model_artifacts/model_best.pt", map_location=device))
model = model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased", use_fast=True)

data = load_dataset(args, "test.csv")
test_dataset = TamilDataset(data, tokenizer, args.labels, args.max_length)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    drop_last=True,
)

test_metrics = model_eval(test_loader, model, args)
os.makedirs('test_results', exist_ok=True)
with open('test_results/test_metrics.json', 'w') as fp:
    json.dump(test_metrics, fp)