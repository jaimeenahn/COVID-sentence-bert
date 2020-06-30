"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""
from torch.utils.data import DataLoader
import math
import torch
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, LabelAccuracyEvaluator
from sentence_transformers.readers import CORD19Reader
import logging
from datetime import datetime
import sys
import pickle

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'
if model_name == 'specter':
    model_path = 'pretrained_models/specter/model.tar.gz'

# You can set your device num
if len(sys.argv) > 2:
    device = sys.argv[2]
else:
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print(device)

data_path = 'data'

# Read the dataset
train_batch_size = 32
num_epochs = 20
model_save_path = 'output/training_cord19'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
cord_reader = CORD19Reader(data_path, normalize_scores=True)

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
if model_name == 'specter':
    # TODO
    train_batch_size = 1
    word_embedding_model = models.Specter(model_path)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=False,
                                   pooling_mode_cls_token=True,
                                   pooling_mode_max_tokens=False)
else:
    if model_name == 'bert-large-uncased':
        # because of OOM
        train_batch_size = 16
    if model_name == 'biobert':
        word_embedding_model = models.BioBERT()
    else:
        word_embedding_model = models.Transformer(model_name)
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

# Convert the dataset to a DataLoader ready for training
logging.info("Read CORD train dataset")
train_data = SentencesDataset(cord_reader.get_examples('qrels-rnd_train.txt'), model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)

# Loss
if model_name == 'bert-large-uncased':
    sentence_embedding_dim = 1024
else:
    sentence_embedding_dim = 768
#train_loss = losses.SoftmaxLoss(model=model,
#        sentence_embedding_dimension=sentence_embedding_dim, num_labels=3)
train_loss = losses.TripleSoftmaxLoss(model=model,
        sentence_embedding_dimension=sentence_embedding_dim, num_labels=3, document_coef=0.2)

logging.info("Read CORD dev dataset")
dev_data = SentencesDataset(examples=cord_reader.get_examples('qrels-rnd_dev.txt'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
evaluator = LabelAccuracyEvaluator(dev_dataloader, device=device, softmax_model=train_loss)


# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_data)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

test_data = SentencesDataset(examples=cord_reader.get_examples("qrels-rnd_test.txt"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=train_batch_size)
test_evaluator = LabelAccuracyEvaluator(test_dataloader, device=device, softmax_model=train_loss)

the_best_val_f1 = -1
the_best_epoch = -1
corresponding_test_f1 = -1
corresponding_test_acc = -1

# Train the model
for i in range(num_epochs):
    best_val_f1, val_acc = model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=1,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

    test_f1, test_acc = model.evaluate(test_evaluator)
    if the_best_val_f1 < best_val_f1:
        the_best_val_f1 = best_val_f1
        the_best_epoch = i
        corresponding_test_f1 = test_f1
        corresponding_test_acc = test_acc
        

print("Best Validation F1 score is {} at {} epoch".format(the_best_val_f1, the_best_epoch))
print("Best Test F1 score is {}".format(corresponding_test_f1))
print("Best Test Accuracy is {}".format(corresponding_test_acc))

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################
#
# test_data = SentencesDataset(examples=cord_reader.get_examples("qrels-rnd_test.txt"), model=model)
# test_dataloader = DataLoader(test_data, shuffle=False, batch_size=train_batch_size)
# evaluator = LabelAccuracyEvaluator(test_dataloader, softmax_model= train_loss)
# model.evaluate(evaluator)
