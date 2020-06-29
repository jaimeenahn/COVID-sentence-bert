import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from ..SentenceTransformer import SentenceTransformer
import logging


class TripleSoftmaxLoss(nn.Module):
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 vocab,
                 doc_model,
                 document_coef: float = 0.4,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False):
        super(TripleSoftmaxLoss, self).__init__()
        self.model = model
        self.doc_model = doc_model
        self.num_labels = num_labels
        self.hidden = 1000
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication
        self.document_coef = document_coef

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 2

        logging.info("Softmax loss: #Vectors concatenated: {}".format(
            num_vectors_concatenated))
        self.document2hidden = nn.Linear(291868, self.hidden)
        self.hidden2output = nn.Linear(self.hidden, 768)
        self.classifier = nn.Linear(
            num_vectors_concatenated * sentence_embedding_dimension, num_labels)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, document_features: Iterable[Dict[str, Tensor]]):
        reps = [self.model(sentence_feature)['sentence_embedding']
                for sentence_feature in sentence_features]
        document_rep = self.model(document_features)['sentence_embedding']
        rep_a, rep_b = reps
        
        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))
            vectors_concat.append(torch.abs(rep_a - document_rep))

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        loss_fct = nn.CrossEntropyLoss()

        if labels is not None:
            loss = loss_fct(output, labels.view(-1))
            # todo: MMI가 들어가면 좋긴하겠다.
            loss -= self.document_coef * \
                torch.sum(torch.cosine_similarity(document_rep, rep_b))
            return loss
        else:
            return reps, output

    def get_doc_model(doc_model):
        self.doc_model = doc_model
