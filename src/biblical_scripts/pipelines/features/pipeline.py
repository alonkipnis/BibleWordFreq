"""
This is a boilerplate pipeline 'features'
generated using Kedro 0.17.0
"""

from kedro.pipeline import Pipeline, node
from .nodes import get_features, plot_features, get_features_chapter

def create_pipeline(**kwargs):
    return Pipeline({node(func=get_features,
                          inputs=["data_proc", "vocabulary", "oshb_parsed",
                                 "params:model", "params:features"],
                          outputs="discriminating_features",
                          name="get_features"
                          ),
                        node(func=get_features_chapter,
                             inputs=["data_proc", 'vocabulary', "oshb_parsed",
                                     "params:model", "params:features"],
                             outputs=None,
                             name="get_features_chapter"
                             ),
                     node(func=plot_features,
                          inputs=["discriminating_features", "params:features"],
                          outputs=None
                          ),
                     }, tags="features"
                    )
