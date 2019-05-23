#-------------------------------------------------------------------------------
# Name:        General Utis
# Purpose:     General utils like pickle load and dump
#
# Author:      jayakrishnan vijayaraghavan,
#
# Created:     19/03/2018
# Licence:     <your licence>
# Copyright:   (c) (c) moveinc 2018
#-------------------------------------------------------------------------------

import pickle

def pickle_load(pickle_file):
    with open( pickle_file, "rb" ) as f:
        d = pickle.load(f)
        print("Pickle loaded from  {}".format(pickle_file))
        return d


def pickle_dump(pyObject, outfile, protocol = 3):
    """Dumps any python object as a pickle file
    Parameters
    ----------
    pyObject: Any Python object
    outfile: Name of the pickle file
    protocol: The pickle protocol to be saved. If pickle is to be read by Python 2.x, protocol = 2 (Default value is 3)

    Returns
    -------
    None
    """
    with open(outfile, 'wb') as f:
        pickle.dump(pyObject, f , protocol)

    print("{} dumped to {}".format("Python Object", outfile))
