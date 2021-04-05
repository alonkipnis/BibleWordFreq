"""
Data Engineering Pipeline. 
"""

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.data_engineering.nodes import (process_data, add_topics, add_convert)

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=process_data, 
             inputs=["oshb_parsed", "params:preprocessing"],
             outputs="data_proc0",
             name="preprocess"
            ),
        node(func=add_topics,
             inputs=["data_proc0", "topics_data"],
             outputs="data_proc1",
             name="topics"
            ),
        node(func=add_convert,
             inputs=["data_proc1", "oshb_parsed"],
             outputs="data_proc",
             name="conversion"
            )
        ]
    )