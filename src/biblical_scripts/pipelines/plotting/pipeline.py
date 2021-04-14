#pipeline: plotting

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.plotting.nodes import (plot_sim, plot_sim_full)

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=plot_sim,
             inputs=["sim_full_res", "params:report", "params:known_authors"],
             outputs=None,
             name="plot_sim"
            ),
        node(func=plot_sim_full,
             inputs=["sim_full_res", "params:report"],
             outputs=None,
             name="plot_sim_full"
            )
        ]
    )
