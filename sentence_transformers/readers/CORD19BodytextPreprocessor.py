import csv
import os
import xml.etree.ElementTree as ET
import json
import pickle
import string
import re
class CORD19BodytextPreprocessor:
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
        self.bodytexts, self.id2bodytext = self.get_bodytext('metadata.csv')

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
        id2query = dict()
        for idx, child in enumerate(root):
            examples.append(child[0].text)
            id2query[child.attrib['number']] = idx
        return examples, id2query

    def get_bodytext(self, filename, max_examples=0):
        filepath = os.path.join(self.dataset_folder, filename)
        with open(filepath, encoding="utf-8") as fIn:
            data = csv.reader(fIn, delimiter=self.delimiter, quoting=self.quoting)
            examples = []
            text_len_list = []
            id2bodytext = dict()
            header = next(data)           

            for idx, row in enumerate(data):
                bodytext = ''
                bodytext_col = len(row) - 3
                bodytext_file_path = row[bodytext_col]
                if len(bodytext_file_path) > 0: # No body text path exists
                     with open(os.path.join(self.dataset_folder, bodytext_file_path)) as bodytext_file:
                        bodytext_dict_list = json.load(bodytext_file)['body_text']    
                        for bodytext_dict in bodytext_dict_list:
                            bodytext += bodytext_dict['text']+" "
                        id2bodytext[row[self.cord_id_idx]] = idx
                        # num_not_empty_text += 1
                examples.append(bodytext)
                text_len_list.append(len(bodytext))
                   
        return examples, id2bodytext

    def save_bodytext(self, paper_id2bodytext, filename):
        filepath = os.path.join(self.dataset_folder, filename)
        with open(filepath, 'rb') as f:
            txt = f.readlines()

        num_bodytext_len = 0
        for idx, text in enumerate(txt):
            content = [t.decode('utf-8') for t in text.split()]
            query = self.queries[self.id2query[content[0]]]
            if (content[2] not in self.id2paper) or (content[2] not in self.id2bodytext):
                continue
            num_bodytext_len += 1
            bodytexts = self.bodytexts[self.id2bodytext[content[2]]]
            paper_id2bodytext[content[2]] = bodytexts
        print(f'{filename}, # valid rows: {num_bodytext_len}, # papers: {len(paper_id2bodytext)}')
        return paper_id2bodytext

    def save_whole_dataset_bodytext(self, dataset_names):
        paper_id2bodytext = dict()
        for dataset_name in dataset_names:
            paper_id2bodytext = self.save_bodytext(paper_id2bodytext, dataset_name)
            # with open('paper_id2bodytext_'+dataset_names[:-4]+'.pkl', 'wb') as f:
            #     pickle.dump(paper_id2bodytext, f)
        with open('paper_id2bodytext_qrels-rnd_dataset.pkl', 'wb') as f:
            pickle.dump(paper_id2bodytext, f)

def open_bodytext_dict(file_name):
    with open(file_name, 'rb') as f:
        pkl_file = pickle.load(f)
    return pkl_file


if __name__ == '__main__':
    cord_reader = CORD19BodytextPreprocessor('/mnt/nas2/seon/ai605_project/COVID_dataset', normalize_scores=True)
    cord_reader.save_whole_dataset_bodytext(['qrels-rnd_train.txt', 'qrels-rnd_dev.txt', 'qrels-rnd_test.txt'])

    # bodytext_dict = open_bodytext_dict('paper_id2bodytext_qrels-rnd_dataset.pkl')
    # print(bodytext_dict, len(bodytext_dict))

