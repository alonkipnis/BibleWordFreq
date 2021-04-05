"""
Module for reading from OSHB. 
OpenBible project inforation: https://hb.openscriptures.org/index.html
Data folder is avaialbe at https://github.com/openscriptures/morphhb/tree/master/wlc
Information on morphological code is available at https://hb.openscriptures.org/parsing/HebrewMorphologyCodes.html

Usage:

Create OSHBData object by providing the path to the local data folder and 
the catalog file containing information on what parts to read. 

"""

from xml.dom.minidom import parse, parseString
from pathlib import Path, PurePosixPath
from kedro.io.core import (
    get_filepath_str,
    get_protocol_and_path,
)

import pandas as pd
from kedro.io import AbstractDataSet
import logging
from tqdm import tqdm


def _read_catalog(catalog_file) :
    """
    The catalog file is a csv file with columns: 'author', 'book', 'chapter', 'verses'
    """
    list_of_ref = pd.read_csv(catalog_file)
    list_of_ref.loc[:,'verses'] = list_of_ref.verses.astype(str)\
                            .apply(lambda vs : vs.strip(";").split(";"))
    list_of_ref = list_of_ref.explode('verses')
    list_of_ref.verses = list_of_ref.verses.replace('nan','all')
    return list_of_ref
    
def _read_from_morph(path, catalog) :
    """
    Read data from OSHB Project according to 'catalog'
    """

    def read_chapter(book, chapter, verse_set = []) :
        df = pd.DataFrame()
        bookxml = parse(path + '/' +book + '.xml')
        chapterlist = bookxml.getElementsByTagName('chapter')
        chapterlist = [ch for ch in chapterlist if ch.attributes['osisID'].value == book+'.'+str(chapter)]
        for chap in chapterlist:
            verselist = chap.getElementsByTagName('verse')
            for verse in verselist:
                mywelements = verse.getElementsByTagName('w')
                for el in mywelements:
                    vrs = verse.attributes['osisID'].value
                    vrs_numeric = int(vrs.split('.')[-1])
                    if len(verse_set) == 0 or vrs_numeric in verse_set :
                        df = df.append({'lemma' : el.attributes['lemma'].value,
                            'morph' : el.attributes['morph'].value,
                            'term' : el.firstChild.data,
                            'chapter' : chap.attributes['osisID'].value,
                            'verse' : vrs
                            }, ignore_index=True)
        return df

    data = pd.DataFrame()
    for c in tqdm(catalog.groupby(['author', 'book', 'chapter'])) :
        for i,r in enumerate(c[1].verses) :
            if r == 'all' :
                vs = []
            else :
                try :
                    a, b = r.split('-')
                except ValueError:
                    a = b = r
                vs = list(range(int(a),int(b)+1))
            logging.debug(f"Reading: author={c[0][0]}, book={c[0][1]}, chapter={c[0][2]}, verse_set={vs}")
            df = read_chapter(book = c[0][1], chapter = c[0][2], verse_set = vs)
            df.loc[:,'author'] = c[0][0]
            data = data.append(df, ignore_index = True)
    return data


class OSHB(AbstractDataSet) :
    """
    Read data from the OSHB project according to the catalog file
    

    """
    def __init__(self, morphhb_path, catalog_file, out_file) :
        self._catalog_file = catalog_file
        self._raw_data_path = morphhb_path
        self._out_file = out_file

        protocol, path = get_protocol_and_path(morphhb_path)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
           
    def _exists(self) -> bool:
        #return Path(self._out_file.as_posix()).exists()
        return False

    def _describe(self):
        return {'raw data' : self._raw_data_path,
                    'catalog file' : self._catalog_file,
                }
        
    def _load(self) -> pd.DataFrame:
        catalog = _read_catalog(self._catalog_file)
        logging.info(f"Found {len(catalog)} entries in catalog file {self._catalog_file}")
         
        logging.info(f"Reading catalog entries from OSHB Project...")
        data = _read_from_morph(self._raw_data_path, catalog)
        return data
                    
    def _save(self, data) -> None: 
        """Saves data to the specified filepath.
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        data.to_csv(save_path)
        