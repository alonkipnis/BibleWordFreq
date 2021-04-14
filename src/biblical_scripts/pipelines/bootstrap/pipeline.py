#pipeline: bootstrap

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.data_science.nodes import filter_by_author
from biblical_scripts.pipelines.bootstrap.nodes import add_BS_stats

from biblical_scripts.pipelines.bootstrap.nodes import (bs_main, bs_main_dist)

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=filter_by_author, 
             inputs=["data_proc", "params:known_authors"],
             outputs="data",
             name="filter_by_author"
            ),
        node(func=bs_main_dist,
             inputs=["data", "params:bootstrapping",
             "vocabulary", "params:model", "params:sim_null", "params:known_authors"],
             outputs="sim_null_res_BS",
            name="sim_null"
            ),
        node(func=add_BS_stats,
        inputs=["sim_null_res_BS1"],
        outputs=["sim_null_res_BS_stats"],
        name="add_stats"
        )
        ]
    )
