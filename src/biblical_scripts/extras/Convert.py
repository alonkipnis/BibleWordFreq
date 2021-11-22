# biblical_scripts.extras

import numpy as np
import pandas as pd
from typing import Tuple, List
import re

class Convert(object):

    """
    Conversions of lemmas to their original form and back
    for data loaded using OSHBDataset

    OSHB Morph codes are in
    https://hb.openscriptures.org/parsing/HebrewMorphologyCodes.html


    Args:
        data    is a dataframe with columns 'lemma', 'morph', 'term. We use
                information in this dataframe to construct a vocabulary
                translating from lemma to term or lemma to morph and back

    """

    def __init__(self, data: pd.DataFrame):

        self._dictionary1 = dict(
            [(k, v) for k, v in zip(data.lemma.values, data.morph.values)])

        self._dictionary = dict(
            [(k, v) for k, v in zip(data.lemma.values, data.term.values)] +
            [('c', 'ו'), ('d', 'ה'), ('b', 'ב'), ('l', 'ל'), ('k', 'כ'),
             ('s', 'ש'), ('i', 'ה')]
                                )

    def _lem2term(self, lemma: str) -> str:
        """
        Convert lemma code (str) to a Hebrew word. Returns lemma code if 
        no matching word is found. 
        """
        dictionary = self._dictionary

        # try special lemmas:
        if re.findall(r"\b854\b", lemma):
            return '(preposition) אֶת'
        if re.findall(r"\b853\b", lemma):
            return '(particle) אֶת'

        return dictionary.get(lemma,  # try different combinations until success,
                                      # if nothing works, return the lemma code
           dictionary.get(lemma + ' a',
           dictionary.get(lemma + ' b',
           dictionary.get(lemma + ' c',
           dictionary.get(lemma + ' d',
           dictionary.get(lemma + ' e',
           dictionary.get(lemma + ' l',
           dictionary.get(lemma + ' m',
           dictionary.get(lemma + ' s',
           dictionary.get('a/'+lemma,
           dictionary.get('b/'+lemma,
           dictionary.get('b/'+lemma + ' a',
           dictionary.get('b/'+lemma + ' b',
           dictionary.get('c/'+lemma,
           dictionary.get('c/'+lemma + ' a',
           dictionary.get('c/'+lemma + ' b',
           dictionary.get('c/'+lemma + ' d',
           dictionary.get('d/'+lemma,
           dictionary.get('d/'+lemma + ' a',
           dictionary.get('d/'+lemma + ' b',
           dictionary.get('d/'+lemma + ' c',
           dictionary.get('e/'+lemma,
           dictionary.get('i/'+lemma,
           dictionary.get('k/'+lemma,
           dictionary.get('l/'+lemma,
           dictionary.get('l/'+lemma + ' a',
           dictionary.get('l/'+lemma + ' b',
           dictionary.get('m/'+lemma,
           dictionary.get('s/'+lemma,
           dictionary.get('s/b/'+lemma,
                    lemma))))))))))))))))))))))))))))))

    def lem2morph(self, lemma: str) -> str:
        return self._dictionary1[lemma]

    def _ng2term(self, ngram: Tuple) -> List:
        """
        Converts a tuple
        """
        return [self._lem2term(l) for l in ngram]

    def to_term(self, token):
        if type(token) == tuple:
            return self._ng2term(token)
        if type(token) == str:
            return self._lem2term(token)