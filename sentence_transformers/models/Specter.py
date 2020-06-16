from torch import nn
from allennlp.models.archival import Archive, load_archive, archive_model
#from allennlp.data.tokenizers import Tokenizer, WordTokenizer
#from allennlp.data.token_indexers import PretrainedBertIndexer
#from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.tokenizers.token import Token
from allennlp.data import Instance, DatasetReader
from allennlp.common.util import import_submodules
from allennlp.data.fields import TextField
from transformers import BertModel, BertTokenizer

from third_parties.specter.specter import SpecterDataReader
#import_submodules("specter.SpecterDataReader")
#import_submodules("specter.Specter")

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
        self.specter.eval() #Archive.extract_module(path=model_name_or_path, freeze=True)
        config = archive.config.duplicate()
        dataset_reader_params = config["dataset_reader"]
        dataset_reader = DatasetReader.from_params(dataset_reader_params)
        self.tokenizer = dataset_reader._tokenizer
        self.token_indexers = dataset_reader._token_indexers
        self.max_seq_length = max_seq_length
        #self.bert = BertModel.from_pretrained(model_name_or_path, **model_args)
        #self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path, **tokenizer_args)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        #print(**features)
        #embed = self.specter.text_field_embedder(features)
        for f in features:
            features[f] = features[f].unsqueeze(0)
        embedded_title, title_mask = self.specter.get_embedding_and_mask(features)
        encoded_title = self.specter.title_encoder(embedded_title, title_mask)
        #output_states = self.specter(**features)
        '''
        print(output_states)
        output_tokens = output_states[0]
        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'attention_mask': features['attention_mask']})

        if len(output_states) > 2:
            features.update({'all_layer_embeddings': output_states[2]})
        '''
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
        #pad_seq_length = min(pad_seq_length, self.max_seq_length) + 2  ##Add Space for CLS + SEP token
        title_field = TextField(tokens, self.token_indexers)
        title_field.index(self.specter.vocab)
        padding_lengths = title_field.get_padding_lengths()
        features = dict()
        features['title'] = title_field.as_tensor(padding_lengths=padding_lengths)
        return title_field.as_tensor(padding_lengths=padding_lengths)
        '''        return self.tokenizer.prepare_for_model(tokens,
        max_length=pad_seq_length, pad_to_max_length=True,
        return_tensors='pt')'''

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}
    
    def save(self, output_path: str):
        archive_model(output_path)
        """self.bert.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)
        """

    @staticmethod
    def load(input_path: str):
        archive = load_archive(input_path)
        self.specter = archive.model
        self.specter.eval()
        return self.specter
    





