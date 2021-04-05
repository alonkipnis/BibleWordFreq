"""
"""

from kedro.io.core import (
    get_filepath_str,
    get_protocol_and_path,
)

import pandas as pd
from kedro.io import AbstractDataSet
import logging


class FigArray(AbstractDataSet) :
    """
    Read data from the OSHB project according to the catalog file
    

    """
    def __init__(self, parameters) :
        self._params = parameters
        self._save_params = {}
           
    def _exists(self) -> bool:
        #return Path(self._out_file.as_posix()).exists()
        return False

    def _describe(self):
        return {'raw data' : self._raw_data_path,
                    'parameters' : self._params,
                }
        
    def _load(self) -> None:
        logging.info(f"This dataset cannot be loaded. ")
                    
    def _save(self, fig) -> None: 
        """Saves data to the specified filepath.
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        fig.save(f"{save_path}",**self._save_params)
        