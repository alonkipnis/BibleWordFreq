#pipeline: Data Engineering 

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.data_engineering.nodes import (process_data, add_topics, add_convert, build_vocab)

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
             name="conversion_proc"
            ),
        node(func=build_vocab, 
             inputs=["data_proc", "params:vocab", "params:known_authors"],
             outputs="vocabulary1",
             name="build_vocab"
            ),
        node(func=add_convert,
             inputs=["vocabulary1", "oshb_parsed"],
             outputs="vocabulary",
             name="conversion_vocab"
            ),
        ]
    )