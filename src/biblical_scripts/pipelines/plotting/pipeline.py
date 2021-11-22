#pipeline: plotting
# 
# uses sim_full_res obtained from sim_full pipeline. Function
# `plot_sim' can also read sim_res

from kedro.pipeline import node, Pipeline
from .nodes import (plot_sim, plot_sim_full)

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=plot_sim,
             inputs=["sim_res", "params:report"],
             outputs=None,
             name="plot_sim"
            ),
        node(func=plot_sim_full,
             inputs=["sim_full_res", "params:report"],
             outputs=None,
             name="plot_sim_full"
            )
        ], tags='plotting'
    )
