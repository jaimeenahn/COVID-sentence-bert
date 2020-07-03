# COVID-sentence-bert

## Installation and Datasets


Link for TF-IDF file: https://drive.google.com/file/d/1_q61oTFFc5IqvvWG5YeRCucTwgrgp2Lq/view?usp=sharing

### Place data
```python
sentence_transformers/readers/CORD19Reader.py
```

## Run

1. Run TF-IDF model

### Choose the model you will run
```python
# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
#BERT
# word_embedding_model = models.Transformer(model_name)
#BioBERT
word_embedding_model = models.BioBERT()
```

### Run
```python
cd examples/training_transformers
python training_cord19.py
```
2. Run BERT model

Change to `jiseon` branch.





