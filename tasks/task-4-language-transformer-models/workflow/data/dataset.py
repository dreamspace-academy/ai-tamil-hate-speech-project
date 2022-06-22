import torch


class TamilDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, labels, max_length):
        self.data = data
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.labels = labels
        self.label2id = {label:id for id,label in enumerate(labels)}

    def __getitem__(self, idx):
        row = self.tokenizer(
            self.data.text.iloc[idx],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors='pt',
            return_token_type_ids=True
        )

        label = torch.LongTensor(
            [self.label2id[self.data.label.iloc[idx]]]
        )

        row['label'] = label

        return row

    def __len__(self):
        return len(self.data)