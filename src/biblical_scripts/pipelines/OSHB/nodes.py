# project: bib-scripts
# pipeline: OSHB
# purpose: extract relevant parts from OSHB Project data according to the catalog

import pandas as pd
import logging

from biblical_scripts.extras.datasets.OSHBDataset import OSHB

def read_OSHB(data) :
    """
    right now this function is here so that Kedro data manager
    would load 'oshb_data'. In the future, we can change this 
    pipline to allow modifications of the catalog (if this catalog is not static)
    """
    return data