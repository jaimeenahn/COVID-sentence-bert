from . import InputExample, Paper, Query
import csv
import logging
import os
import xml.etree.ElementTree as ET
import pickle

class CORD19Reader:
    """
    Reads in the CORD dataset. Each file contains [cord_uid, sha,source_x,title,doi, pmcid, pubmed_id, license,
    abstract, publish_time, authors, journal, mag_id,who_covidence_id, arxiv_id, pdf_json_files, pmc_json_files, url,
     s2_id]

    line contains two sentences (s1_col_idx, s2_col_idx) and one label (score_col_idx)

    Default values expects a tab seperated file with the first & second column the sentence pair and third column the
    score (0...1). Default config normalizes scores from 0...5 to 0...1
    """
    def __init__(self, dataset_folder, cord_id_col=0, title_col=3, abstract_col=8, delimiter=",",
                 quoting=csv.QUOTE_NONE, normalize_scores=True, min_score=0, max_score=5):
        self.dataset_folder = dataset_folder
        self.abstract_col = abstract_col
        self.cord_id_idx = cord_id_col
        self.title_col = title_col
        self.delimiter = delimiter
        self.quoting = quoting
        self.normalize_scores = normalize_scores
        self.min_score = min_score
        self.max_score = max_score
        self.papers, self.id2paper = self.get_papers('metadata.csv')
        self.queries, self.id2query = self.get_queries('topics-rnd2.xml')
        self.bodytexts, self.id2bodytext = self.get_bodytexts('paper_id2bodytext_qrels-rnd_dataset.pkl')

    def get_papers(self, filename, max_examples=0):
        filepath = os.path.join(self.dataset_folder, filename)
        with open(filepath, encoding="utf-8") as fIn:
            data = csv.reader(fIn, delimiter=self.delimiter, quoting=self.quoting)
            examples = []
            id2paper = dict()
            header = next(data)
            for idx, row in enumerate(data):

                s1 = row[self.title_col]
                s2 = row[self.abstract_col]

                txt = s1.strip()+' [SEP] ' + s2.strip()
                examples.append(txt)
                id2paper[row[self.cord_id_idx]] = idx
                if max_examples > 0 and len(examples) >= max_examples:
                    break

        return examples, id2paper

    def get_queries(self, filename, max_examples=0):

        filepath = os.path.join(self.dataset_folder, filename)
        tree = ET.parse(filepath)
        root = tree.getroot()
        examples = []
        id2query = {}
        for idx, child in enumerate(root):
            examples.append(child[0].text)
            id2query[child.attrib['number']] = idx
        return examples, id2query

    def get_examples(self, filename, max_input_length):
        """
        filename specified which data split to use (qrels-rnd#_train.csv, qrels-rnd#_dev.csv, qrels-rnd#_test.csv).
        """
        filepath = os.path.join(self.dataset_folder, filename)
        with open(filepath, 'rb') as f:
            txt = f.readlines()
        examples = []
        for idx, text in enumerate(txt):
            content = [t.decode('utf-8') for t in text.split()]
            query = self.queries[self.id2query[content[0]]]
            if content[2] not in self.id2paper:
                continue
            if content[2] not in self.id2bodytext:
                continue

            paper = self.papers[self.id2paper[content[2]]]
            label = int(content[3])

            bodytext = " ".join(self.bodytexts[self.id2bodytext[content[2]]].split(" ")[:max_input_length])

            examples.append(InputExample(guid=idx, texts=[query, paper], label=label, bodytext=bodytext))

        return examples

    def get_bodytexts(self, filename):
        logging.info("Read bodytext...")
        with open('/mnt/nas2/seon/ai605_project/COVID_dataset/'+filename, 'rb') as f:
            idx_bodytext_dict = pickle.load(f)

        bodytexts = []
        id2bodytext = {}
        for idx, bodytext_idx in enumerate(idx_bodytext_dict):
            bodytexts.append(idx_bodytext_dict[bodytext_idx])
            id2bodytext[bodytext_idx] = idx
            
        return bodytexts, id2bodytext
