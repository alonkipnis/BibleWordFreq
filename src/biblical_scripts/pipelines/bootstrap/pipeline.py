#pipeline: bootstrap

"""
The idea here is to create many bootstrap dataset by sampling
n-grams with repetations and evalaute the performance of attribution
on each one. 
"""

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.sim.nodes import filter_by_author
from biblical_scripts.pipelines.reporting.nodes import add_stats_BS
from biblical_scripts.pipelines.plotting.nodes import (plot_sim_bs, plot_sim_full_bs)

from .nodes import (bs_main_val, bs_main_full_dist, bs_main_full_dist)


def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=filter_by_author, 
             inputs=["data_proc", "params:known_authors"],
             outputs="data",
             name="filter_by_author"
            ),
        node(func=bs_main_val,
             inputs=["data", "params:bootstrapping",
             "vocabulary", "params:model", "params:known_authors"],
             outputs="sim_res_BS",
            name="bs_main_val"
            ),
        # node(func=bs_main_full,
        #      inputs=["data", "params:bootstrapping",
        #      "vocabulary", "params:model", "params:sim_full", "params:known_authors"],
        #      outputs="sim_full_res_BS",
        #     name="sim_full"
        #     ),
        node(func=add_stats_BS,
            inputs=["sim_res_BS", "params:report"],
            outputs="sim_res_BS_stats",
            name="add_stats_BS"
            ),
        node(func=plot_sim_bs,
             inputs=["sim_res_BS_stats", "params:report", "params:known_authors"],
             outputs=None,
             name="plot_sim_bs"
             ),
        # node(func=plot_sim_full_bs,
        #      inputs=["sim_res_BS_stats", "params:report"],
        #      outputs=None,
        #      name="plot_sim_full_bs"
        #      )
        ], tags='bootstrap'
    )
