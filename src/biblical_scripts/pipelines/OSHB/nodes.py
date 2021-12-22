# project: bib-scripts
# pipeline: OSHB
# purpose: extract relevant parts from OSHB Project data according to the catalog

import pandas as pd
import logging

from biblical_scripts.extras.datasets.OSHBDataset import OSHB

def read_OSHB(data) :
    """
    Right now this function is here so that Kedro data manager
    would load 'oshb_data'.

    If this catalog were not static, we could change this pipeline
     accordingly.

    """
    return data