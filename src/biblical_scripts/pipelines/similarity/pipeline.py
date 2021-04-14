#pipeline: bootstrapping

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.data_science.nodes import (build_reduced_vocab, evaluate_accuracy, filter_by_author)

from biblical_scripts.pipelines.data_science_val.nodes import (cross_validation)
from biblical_scripts.pipelines.plotting.nodes import plot_sim
from biblical_scripts.pipelines.similarity.nodes import sim_null

from biblical_scripts.pipelines.plotting.nodes import plot_sim_null

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=filter_by_author, 
             inputs=["data_proc", "params:known_authors"],
             outputs="data",
             name="filter_by_author"
            ),
        node(func=sim_null,
             inputs=["data", "vocabulary", "params:model", "params:sim_null", "params:known_authors"],
             outputs="sim_null_res",
            name="sim_null"
            )
        ]
    )
