import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from ..SentenceTransformer import SentenceTransformer

class Concat(nn.Module):
    def __init__(self, model: SentenceTransformer):
        super(Concat, self).__init__()
        self.model = model
        self.linear = nn.Linear(2*768, 3)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps
        output = self.linear(torch.cat([rep_a, rep_b], 1))

        loss_fct = nn.CrossEntropyLoss()

        if labels is not None:
            loss = loss_fct(output, labels.view(-1))
            return loss
        else:
            return reps, output