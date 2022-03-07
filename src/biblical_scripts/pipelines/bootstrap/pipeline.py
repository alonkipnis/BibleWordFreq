#pipeline: bootstrap

"""
The idea here is to create many bootstrap dataset by sampling
n-grams with repetations and evalaute the performance of attribution
on each one. 
"""

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.sim.nodes import filter_by_author
from biblical_scripts.pipelines.reporting.nodes import add_stats_BS

from .nodes import bs_main_val


def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=filter_by_author,
             inputs=["data_proc", "params:all_authors",
                     "params:unknown_authors", "params:only_reportables"],
             outputs="data",
             name="filter_by_author"
            ),
        node(func=bs_main_val,
             inputs=["data", "params:bootstrapping",
             "vocabulary", "params:model", "params:known_authors"],
             outputs="sim_res_BS",
            name="bs_main_val"
            ),
        node(func=add_stats_BS,
            inputs=["sim_res_BS", "params:report"],
            outputs="sim_res_BS_stats",
            name="add_stats_BS"
            ),
        ], tags='bootstrap'
    )

