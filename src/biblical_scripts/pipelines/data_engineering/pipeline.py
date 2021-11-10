#pipeline: Data Engineering 

from kedro.pipeline import node, Pipeline
from .nodes import (process_data, add_topics, add_convert, build_vocab)

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=process_data, 
             inputs=["oshb_parsed", "params:preprocessing"],
             outputs="data_proc0",
            ),
        #node(func=add_topics,
        #     inputs=["data_proc0", "topics_data"],
        #     outputs="data_proc1",
        #    ),
        node(func=add_convert,
             inputs=["data_proc0", "oshb_parsed"],
             outputs="data_proc",
            ),
        node(func=build_vocab, 
             inputs=["data_proc", "params:vocab", "params:known_authors"],
             outputs="vocabulary1",
            ),
        node(func=add_convert,
             inputs=["vocabulary1", "oshb_parsed"],
             outputs="vocabulary",
            ),
        ], tags='data engineering'
    )