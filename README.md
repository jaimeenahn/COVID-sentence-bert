# COVID-sentence-bert

## Installation and Datasets

Link for body_text file: https://drive.google.com/file/d/1hCXeldRJLC3zAZduN46lnah5b-31RTlz/view?usp=sharing

### Place data
```python
sentence_transformers/readers/CORD19Reader.py
```

## Model: Lonformer
Description for Longformer: https://huggingface.co/transformers/model_doc/longformer.html

![alt text](https://github.com/jaimeenahn/COVID-sentence-bert/blob/jiseon/longformer_overview.png?raw=true)

## How to Run

### Change max batch_size/epoch/max_input_length

in examples/training_transformers/training_cord19.py
```python
train_batch_size = 2
num_epochs = 20
max_input_length = 1000 # for fast preprocessing
```

### Choose the model you will run
```python
# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
#BERT
word_embedding_model = models.Transformer(model_name)
```

### Run
```python
cd examples/training_transformers
python training_cord19.py
```
