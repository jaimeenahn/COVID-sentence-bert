from torch import nn
from allennlp.models.archival import Archive, load_archive, archive_model
from allennlp.data.tokenizers.token import Token
from allennlp.data import Instance, DatasetReader
from allennlp.common.util import import_submodules
from allennlp.data.fields import TextField
from transformers import BertModel, BertTokenizer

from third_parties.specter.specter import SpecterDataReader

import json
from typing import List, Dict, Optional
import os
import numpy as np
import logging

class Specter(nn.Module):
    """Specter model to generate token embeddings.

    Each token is mapped to an output vector from BERT.
    """
    def __init__(self, model_name_or_path: str, max_seq_length: int = 200,
            do_lower_case: bool = True, model_args: Dict = {}, tokenizer_args: Dict = {}):
        super(Specter, self).__init__()
        
        archive = load_archive(model_name_or_path)
        self.specter = archive.model
        self.specter.eval()
        config = archive.config.duplicate()
        dataset_reader_params = config["dataset_reader"]
        dataset_reader = DatasetReader.from_params(dataset_reader_params)
        self.tokenizer = dataset_reader._tokenizer
        self.token_indexers = dataset_reader._token_indexers

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        # TODO Currently, it only works in batch size 1.
        for f in features:
            features[f] = features[f].unsqueeze(0)
        embedded_title, title_mask = self.specter.get_embedding_and_mask(features)
        encoded_title = self.specter.title_encoder(embedded_title, title_mask)
        features.update({'cls_token_embeddings': encoded_title, 'token_embeddings': None, 'attention_mask': None})
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.specter.feedforward.get_output_dim()

    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        if ' [SEP] ' in text:
            title, abstract = text.split(' [SEP] ')
            return self.tokenizer.tokenize(title) + [Token('[SEP]')] + self.tokenizer.tokenize(abstract)
        else:
            return self.tokenizer.tokenize(text)

    def get_sentence_features(self, tokens: List[Token], pad_seq_length: int):
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask

        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        title_field = TextField(tokens, self.token_indexers)
        title_field.index(self.specter.vocab)
        padding_lengths = title_field.get_padding_lengths()
        features = dict()
        features['title'] = title_field.as_tensor(padding_lengths=padding_lengths)
        return title_field.as_tensor(padding_lengths=padding_lengths)

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}
    
    def save(self, output_path: str):
        archive_model(output_path)

    @staticmethod
    def load(input_path: str):
        archive = load_archive(input_path)
        self.specter = archive.model
        self.specter.eval()
        return self.specter
    





