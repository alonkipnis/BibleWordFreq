"""
OSHB Pipeline. 
"""

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.OSHB.nodes import read_OSHB

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=read_OSHB,
             inputs=['oshb_raw'],
             outputs='oshb_parsed',
             name="OSHB_reader"
            ),
        ]
    )