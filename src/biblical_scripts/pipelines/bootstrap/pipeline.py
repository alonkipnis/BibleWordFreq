#pipeline: bootstrap

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.data_science.nodes import filter_by_author
from biblical_scripts.pipelines.report.nodes import add_stats_BS

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
             "vocabulary", "params:model", "params:sim_full", "params:known_authors"],
             outputs="sim_full_res_BS",
            name="sim_full"
            ),
        node(func=add_stats_BS,
        inputs=["sim_full_res_BS1"],
        outputs=["sim_full_res_BS_stats"],
        name="add_stats"
        )
        ]
    )
