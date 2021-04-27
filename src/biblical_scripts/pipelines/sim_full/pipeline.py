#pipeline: sim full

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.sim.nodes import (evaluate_accuracy, filter_by_author)

from .nodes import sim_full

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=filter_by_author, 
             inputs=["data_proc", "params:all_authors", "params:unk_authors"],
             outputs="data",
             name="filter_by_author"
            ),
        node(func=sim_full,
             inputs=["data", "vocabulary", "params:model", "params:sim_full", "params:known_authors"],
             outputs="sim_full_res",
            name="sim_full"
            )
        ], tags='full similarity'
    )
