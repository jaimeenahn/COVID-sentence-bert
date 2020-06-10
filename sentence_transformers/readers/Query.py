from typing import Union, List


class Query:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, qid: str, texts: List[str]):
        """
        Creates one InputExample with the given texts, guid and label

        str.strip() is called on both texts.

        :param guid
            id for the example
        :param texts
            the texts for the example
        :param label
            the label for the example
        """
        self.qid = qid
        self.texts = texts.strip()

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))