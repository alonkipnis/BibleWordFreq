"""
OSHB Pipeline. 
"""

from kedro.pipeline import node, Pipeline
from .nodes import read_OSHB

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=read_OSHB,
             inputs=['oshb_raw'],
             outputs='oshb_parsed',
             name="OSHB_reader"
            ),
        ], tags="raw data parser"
    )