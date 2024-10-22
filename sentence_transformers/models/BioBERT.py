from torch import nn
from transformers import BertModel, BertTokenizer, BertConfig
import json
from typing import List, Dict, Optional
import os
import torch
from collections import OrderedDict
import numpy as np
import logging

class BioBERT(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.
    """
    def __init__(self, max_seq_length: int = 128, model_args: Dict = {}, cache_dir: Optional[str] = None ):
        super(BioBERT, self).__init__()
        self.config_keys = ['max_seq_length']
        self.max_seq_length = max_seq_length

        config = BertConfig.from_json_file('/mnt/nas2/jaimeen/COVID/BioBERT/config.json')
        self.auto_model = BertModel(config=config)
        self.vocab = self.load_bert_vocab('/mnt/nas2/jaimeen/COVID/BioBERT/vocab.txt')
        self.tokenizer = BertTokenizer(vocab_file='/mnt/nas2/jaimeen/COVID/BioBERT/vocab.txt', max_length=max_seq_length)

    def load_bert_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = OrderedDict()
        index = 0
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab

    def load_pretrained(self, config):
        state_dict = torch.load('/mnt/nas2/jaimeen/COVID/BioBERT/pytorch_model.bin')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('bert.'):
                k = k.replace('bert.', '')
                new_state_dict[k] = v
            elif k.startswith('cls.'):
                continue
            else:
                new_state_dict[k] = v
        self.model = BertModel(config)
        self.model.load_state_dict(new_state_dict)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        output_states = self.auto_model(**features)
        output_tokens = output_states[0]

        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'attention_mask': features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def get_sentence_features(self, tokens: List[int], pad_seq_length: int):
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask

        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        pad_seq_length = min(pad_seq_length, self.max_seq_length) + 3 #Add space for special tokens
        return self.tokenizer.prepare_for_model(tokens, max_length=pad_seq_length, pad_to_max_length=True, return_tensors='pt')

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, '/mnt/nas2/jaimeen/COVID/BioBERT/config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)






