import torch
from torch.utils.data import DataLoader
from collections import Counter
import functools
import os
import pandas as pd
from .dataset import TamilDataset


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


def get_labels_frequencies(train_dataset):
    label_freqs = Counter(train_dataset['label'].tolist())
    return label_freqs


def load_dataset(args, file_name):
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
    data_path = os.path.join(data_path, file_name)
    data = pd.read_csv(data_path, index_col=0, header=0)
    return data


def get_data_loaders(args, train_corpus, dev_corpus, tokenizer):

    args.label_freqs = get_labels_frequencies(train_corpus)
    args.num_labels = len(args.labels)

    train_dataset = TamilDataset(train_corpus, tokenizer, args.labels, args.max_length)
    args.train_data_len = len(train_corpus)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_dataset = TamilDataset(dev_corpus, tokenizer, args.labels, args.max_length)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
