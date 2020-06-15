from typing import Union, List


class Paper:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, pid: str, texts: List[str]):
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
        self.pid = pid
        self.texts = texts

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))