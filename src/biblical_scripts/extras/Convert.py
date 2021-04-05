# biblical_scripts.extras

import numpy as np
import pandas as pd
from typing import Tuple, List

class Convert :
    """
    class to provide conversions of lemma to original terms and back
    for data loaded using OSHBDataset
    
    data is a data frame with columns 'lemma', 'morph', 'term'
    
    """
    def __init__(self, data : pd.DataFrame) :
        
        self._dictionary1 = dict([(k,v) for k,v in zip(data.lemma.values, data.morph.values)])
    
        self._dictionary = dict([(k,v) for k,v in zip(data.lemma.values, data.term.values)]
                  + [('c','ו'),('d','ה'), ('b','ב'), ('l', 'ל'), ('k', 'כ')])

    def lem2term(self, lemma : str) -> str :
        """
        Convert lemma code (str) to a Hebrew word. Returns lemma code if 
        no matching word is found. 
        """
        dictionary = self._dictionary
        return dictionary.get(lemma, 
           dictionary.get(lemma + ' a',
           dictionary.get(lemma + ' b',
           dictionary.get(lemma + ' c',
           dictionary.get(lemma + ' d',
           dictionary.get(lemma + ' e',
           dictionary.get(lemma + ' l',
           dictionary.get(lemma + ' m',
           dictionary.get('a/'+lemma,
           dictionary.get('b/'+lemma,
           dictionary.get('c/'+lemma,
           dictionary.get('c/'+lemma + ' b',
           dictionary.get('d/'+lemma,
           dictionary.get('e/'+lemma,
           dictionary.get('k/'+lemma,
           dictionary.get('l/'+lemma,
           dictionary.get('m/'+lemma,
           dictionary.get('m/'+lemma,
                    lemma))))))))))))))))))
    
    def lem2morph(self, lemma : str) -> str :
        return self._dictionary1[lemma]
        
    def ng2term(self, ngram : Tuple) -> List :
        """
        Converts a tuple
        """
        return [self.lem2term(l) for l in ngram]