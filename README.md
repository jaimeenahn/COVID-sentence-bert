# COVID-sentence-bert

## Installation and Datasets

Link for BioBERT:

Link for body_text file: https://drive.google.com/file/d/1hCXeldRJLC3zAZduN46lnah5b-31RTlz/view?usp=sharing

### Place data
```python
sentence_transformers/readers/CORD19Reader.py
```

## Run

### Choose the model you will run
```python
# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
#BERT
# word_embedding_model = models.Transformer(model_name)
#BioBERT
# word_embedding_model = models.BioBERT()
word_embedding_model = models.Longformer()
```

### Run
```python
cd examples/training_transformers
python training_cord19.py
```






