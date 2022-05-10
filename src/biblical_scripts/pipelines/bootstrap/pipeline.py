#pipeline: bootstrap

"""
The idea here is to create many bootstrap dataset by sampling
n-grams with repetations and evalaute the performance of attribution
on each one. 
"""

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.sim.nodes import filter_by_author
from biblical_scripts.pipelines.reporting.nodes import (
    add_stats_BS, add_stats_BS_full, comp_probs,
    summarize_probs_BS)

from .nodes import bs_main_val, bs_main_full

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=filter_by_author,
             inputs=["data_proc", "params:all_authors",
                     "params:unknown_authors", "params:only_reportables"],
             outputs="data",
             name="filter_by_author"
            ),
        node(func=bs_main_full,
             inputs=["data", "params:bootstrapping",
             "params:vocab", "params:model", "params:sim_full",
                     "params:known_authors", "params:report",
                     "reference_data"],
             outputs="sim_res_BS",
            name="bs_main_val"
            ),
        # node(func=add_stats_BS,
        #     inputs=["sim_res_BS", "params:report"],
        #     outputs="sim_res_BS_stats",
        #     name="add_stats_BS"
        #     ),
        #     node(func=add_stats_BS_full,
        #          inputs=["sim_res_BS", "params:report"],
        #          outputs="probs_BS",
        #          name="add_stats_BS_full"
        #          ),
        #     node(func=comp_probs,
        #          inputs=["sim_res_BS", "params:report"],
        #          outputs="probs_BS",
        #          name="comp_probs"
        #          ),
            # node(func=summarize_probs_BS,
            #      inputs=["probs_BS", "params:report"],
            #      outputs="summary",
            #      name="summarize_probs_BS"
            #      ),
        ], tags='bootstrap'
    )

