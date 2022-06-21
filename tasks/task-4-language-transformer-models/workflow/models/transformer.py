import torch.nn as nn
from transformers import AutoConfig, AutoModel, DistilBertModel


class TxtEncoder(nn.Module):
    def __init__(self, args):
        super(TxtEncoder, self).__init__()
        self.args = args
        config = AutoConfig.from_pretrained(
            args.model_path,
        )
        self.model = AutoModel.from_pretrained(
            args.model_path,
            config=config,
        )

    def forward(self, txt, mask, segment=None):
        if segment is not None and not isinstance(self.model, DistilBertModel):
            out = self.model(
                input_ids=txt,
                token_type_ids=segment,
                attention_mask=mask,
            )
        else:
            out = self.model(
                input_ids=txt,
                attention_mask=mask,
            )

        return out.last_hidden_state[:,0,:]

class SentClf(nn.Module):
    def __init__(self, args):
        super(SentClf, self).__init__()
        num_outputs = 1 if args.num_labels == 2 else args.num_labels
        self.args = args
        self.enc = TxtEncoder(args)
        self.clf = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.BatchNorm1d(args.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size, num_outputs)
        )

    def forward(self, txt, mask, segment=None):
        x = self.enc(txt, mask, segment)
        return self.clf(x)
