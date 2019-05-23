#-------------------------------------------------------------------------------
# Name:        geomUtils
# Purpose:     Common geometry Utility functions
#
# Author:      realtor.com, jayakrishnan vijayaraghavan, 
#
# Created:     19/03/2018
# Licence:     <your licence>
# Copyright:   (c) (c) moveinc 2018
#-------------------------------------------------------------------------------
import numpy as np

"""
DO NOT CHANGE THIS VALUE UNLESS THE RADIUS OF THE EARTH CHANGES !!!
"""

"""
Conversion Values
"""
RADIUS_OF_EARTH = 6371.0088 ##In kilometers; Change this to miles if you don't believe in metric system
KMS_TO_METERS = 1000.0
METERS_TO_KILOMETERS = 0.001
MILES_TO_KILOMETERS = 1.60934
KILOMETERS_TO_MILES = 0.621371



def gcd(loc1, loc2, R=RADIUS_OF_EARTH * KMS_TO_METERS):
    """Great circle distance via Haversine formula
    Parameters
    ----------
    loc1: tuple (long, lat in decimal degrees)
    loc2: tuple (long, lat in decimal degrees)
    R: Radius of the earth (3961 miles, 6367 km)

    Returns
    -------
    great circle distance between loc1 and loc2 in units of R

    Notes
    ------
    Does not take into account non-spheroidal shape of the Earth
    >>> san_diego = -117.1611, 32.7157
    >>> austin = -97.7431, 30.2672
    >>> gcd(san_diego, austin)
    1155.474644164695

    """
    lon1, lat1 = loc1
    lon2, lat2 = loc2
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
    c = 2*np.arcsin(np.sqrt(a))

    return R * c

def compareFloat(a, b, rTol = .00001, aTol = .00000001):
    """Uses the same formula numpy's allclose function:

    INPUTS:
    a (float): float to be compared
    b (float): float to be compared
    rTol (float): relative tolerance
    aTol (float): absolute tolerance

    OUTPUT:
    return (boolean): true if |a - b| < aTol + (rTol * |b|)
    """

    try:
        if abs(a - b) < aTol + (rTol * abs(b)):
            return True
        else:
            return False
    except:
        return False


def convert2Radians(degree):
    """Converts degree to radians.

    INPUTS:
    degree (float): degree of angle

    RETURN (float): radians of angle
    """
    return np.pi / 180.0 * degree

def convert2degrees(radians):
    """Converts degree to radians.

    INPUTS:
    radians (float): radians of angle

    RETURN (float): degree of angle
    """
    return 180.0 / np.pi * radians
