#pipeline: plotting_BS

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.plotting.nodes import (plot_sim_BS, plot_sim_full_BS)

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=plot_sim_BS,
             inputs=["sim_full_res_BS_stats", "params:report", "params:known_authors"],
             outputs=None,
             name="plot_sim_BS"
            ),
        node(func=plot_sim_full_BS,
             inputs=["sim_full_res_BS_stats", "params:report"],
             outputs=None,
             name="plot_sim_full_BS"
            )
        ]
    )
