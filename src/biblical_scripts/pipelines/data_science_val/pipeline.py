#pipeline: data science val 

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.data_science.nodes import (build_reduced_vocab, evaluate_accuracy, filter_by_author)

from biblical_scripts.pipelines.data_science_val.nodes import (cross_validation)
from biblical_scripts.pipelines.plotting.nodes import plot_sim

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=filter_by_author, 
             inputs=["data_proc", "params:known_authors"],
             outputs="data",
             name="filter_by_author"
            ),
        node(func=cross_validation,
             inputs=["data", "vocabulary", "params:model", "params:report"],
             outputs="cross_val"
            ),
        node(func=plot_sim,
             inputs=["cross_val", "params:report", "params:known_authors"],
             outputs="",
             name="illustrate"
            )
        ]
    )
