#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      realtor.com, jayakrishnan vijayaraghavan,
#
# Created:     19/03/2018
# Licence:     <your licence>
# Copyright:   (c) (c) moveinc 2018
#-------------------------------------------------------------------------------
import numpy as np
import geomutils as gu

def CalculateStandardCircleParams(w, xy, min_radius = 5.0):
    try:
        numFeatures = len(w)
        w.shape = numFeatures, 1
        xyWeighted = w * xy
        weightSum = w.sum()

        #### Mean Center ####
        centers = xyWeighted.sum(0) / weightSum
        distances = [gu.gcd(centers, xy_item, gu.RADIUS_OF_EARTH) for xy_item in xy]
        avg_dist = np.mean(distances)
        std_dev_dist = np.std(distances)
        _1std_dev_dist = avg_dist + std_dev_dist
        radius = _1std_dev_dist if _1std_dev_dist > min_radius else min_radius

        if centers  is None or radius is  None:
            raise ValueError('radius and/or center are null')
        return {"shape" : "circle", "parameters" : {"distance" : radius}, "centroid" : centers}
    except Exception as e:
        raise Exception(e)


def CalculateStandardEllipseParams(w, xy, stdDeviations):
    """Calculates the Standard Deviational Directional Ellipse.

    INPUTS:
    visid (numpy array of string): Userid

    w (numpy array float): weights of feature

    xy (numpy array of coordinates): Coordinate array

    RETURN (tuple): (Centers, semi-major axis, semi-minor axis,
    Degree of Rotation, RadianRotation1, radianRotation2)
    """
    try:
        numFeatures = len(w)
        w.shape = numFeatures, 1
        xyWeighted = w * xy
        weightSum = w.sum()

        #### Mean Center ####
        centers = xyWeighted.sum(0) / weightSum
        devXY = xy - centers
        flatW = w.flatten()
        sigX = (flatW * devXY[:,0]**2.0).sum()
        sigY = (flatW * devXY[:,1]**2.0).sum()
        sigXY = (flatW * devXY[:,0] * devXY[:,1]).sum()
        denom = 2.0 * sigXY
        diffXY = sigX - sigY
        sum1 = diffXY**2.0 + 4.0 * sigXY**2.0

        if not abs(denom) > 0:
            arctanVal = 0.0
        else:
            tempVal = (diffXY + np.sqrt(sum1)) / denom
            arctanVal = np.arctan(tempVal)

        if arctanVal < 0.0:
            arctanVal += (np.pi / 2.0)

        sinVal = np.sin(arctanVal)
        cosVal = np.cos(arctanVal)
        sqrt2 = np.sqrt(2.0)
        sigXYSinCos = 2.0 * sigXY * sinVal * cosVal
        seA = sqrt2 * np.sqrt(((sigX * cosVal**2.0) - sigXYSinCos + (sigY * sinVal**2.0))/weightSum) * stdDeviations
        seB = sqrt2 * np.sqrt(((sigX * sinVal**2.0) + sigXYSinCos + (sigY * cosVal**2.0)) /weightSum) * stdDeviations

        #### Counter Clockwise from Noon ####
        degreeRotation = 360.0 - (arctanVal * 57.2957795)

        #### Convert to Radians ####
        radianRotation1 = gu.convert2Radians(degreeRotation)

        #### Add Rotation ####
        radianRotation2 = 360.0 - degreeRotation
        if seA > seB:
            radianRotation2 += 90.0
            if radianRotation2 > 360.0:
                radianRotation2 = radianRotation2 - 180.0
        if centers is None or seA is None:
            raise ValueError('ellipse variables are null')
        ellipse_params = {"shape" : "standard_ellipse", "centroid" : centers, \
        "parameters" : {"seA" : seA, "seB" :seB, "degreeRotation" :degreeRotation, \
        "radianRotation1" : radianRotation1, "radianRotation2" : radianRotation2}}
        return ellipse_params
    except Exception as e:
        raise ValueError("Error in Calculating Standard Ellipse Params")


def GenerateEllipseCoordinates(se, step = 18, min_radius = 5.0):

    centers = se["centroid"]

    if True in np.isnan(centers):
        raise ValueError('Invalid centroid')

    xVal, yVal = centers
    poly = []
    errors = set()
    if se["shape"] == "standard_ellipse":

        seA = se["parameters"]["seA"]
        seB = se["parameters"]["seB"]
        degreeRotation = se["parameters"]["degreeRotation"]
        radianR1 = se["parameters"]["radianRotation1"]
        radianR2 = se["parameters"]["radianRotation2"]
        
        if seA == 0 or seB == 0:
            seA = seB = np.radians(min_radius)

        ratio = seA/seB
        
        if ratio > 3.0:
            ratio = 3.0
        elif ratio < 1/3.0:
            ratio = 1/3.0
        seA = ratio * seB

        seA2 = seA**2.0
        seB2 = seB**2.0
        

        #### Check for Valid Radius ####
        seAZero = gu.compareFloat(0.0, seA, rTol = .0000001)
        seANan = np.isnan(seA)
        seABool = seAZero + seANan

        seBZero = gu.compareFloat(0.0, seB, rTol = .0000001)
        seBNan = np.isnan(seB)
        seBBool = seBZero + seBNan

        if not seABool and not seBBool:
            cosRadian = np.cos(radianR1)
            sinRadian = np.sin(radianR1)

        #### Calculate a Point For Each ####
        #### Degree in Ellipse Polygon ####
        for degree in np.arange(0, 360, step):

            radians = gu.convert2Radians(degree)
            tanVal2 = np.tan(radians)**2.0
            try:
                dX = np.sqrt((seA2 * seB2) /(seB2 + (seA2 * tanVal2)))
                dY = 0 if seA2 - dX**2 < 0   else np.sqrt((seB2 * (seA2 - dX**2.0)) /seA2)

                #### Adjust for Quadrant ####
                if 90 <= degree < 180:
                    dX = -dX
                elif 180 <= degree < 270:
                    dX = -dX
                    dY = -dY
                elif degree >= 270:
                    dY = -dY

                #### Rotate X and Y ####
                dXr = dX * cosRadian - dY * sinRadian
                dYr = dX * sinRadian + dY * cosRadian
                #### Create Point Shifted to ####
                #### Ellipse Centroid ####
                pntX = dXr + xVal
                pntY = dYr + yVal

                poly.append([pntX, pntY])

            except Exception as e:
                #Continue generating other coordinates
                errors.add(e)
                continue

    else: #Create a Circle
        radius = se["parameters"]["distance"]
        #### Calculate a Point For Each ####
        #### Degree in Circle Polygon ####

        for degree in np.arange(0, 360, step):
            try:
                bearing = gu.convert2Radians(degree)
                delta = radius / gu.RADIUS_OF_EARTH
                lat = gu.convert2Radians(yVal)
                lon = gu.convert2Radians(xVal)
                new_lat = np.arcsin(np.sin(lat) * np.cos(delta) + np.cos(lat) * np.sin(delta) * np.cos(bearing))
                new_lon = lon + np.arctan2(np.sin(bearing) * np.sin(delta) * np.cos(lat) ,  np.cos(delta) - np.sin(lat) * np.sin(new_lat))
                pntX = gu.convert2degrees(new_lon)
                pntY = gu.convert2degrees(new_lat)
                poly.append([pntX, pntY])

            except Exception as e:
                print(e)
                errors.add(e)
                #Continue Generating other coordinates
                continue

    if len(poly) < 10:
        raise ValueError("Error generating coordinates. Insufficient points generated to create geometry" + '. '.join(errors))
    poly.append(poly[0])
    return poly
