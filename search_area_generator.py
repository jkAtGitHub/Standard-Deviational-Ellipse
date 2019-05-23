#-------------------------------------------------------------------------------
# Name:        SEARCH_AREA_GENERATOR
# Purpose:     Generates Search Area for each User
#
# Author:      jayakrishnan vijayaraghavan,
#
# Created:     23/03/2018
# Copyright:   (c) moveinc 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import geomutils as gu
import ellipsegen as ellipse
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

class LocationObj(object):
    
    def __init__(self, 
                 visid,
                 coords,
                 weights, 
                 city_list, 
                 state_list, 
                 stdDeviations = 2.5,  
                 geom_cnt = 3, 
                 num_ellipse_coords = 20, 
                 min_radius = 5):
        
        self.identifier = visid
        self.coords = coords
        self.weights = weights
        self.city_list = city_list
        self.state_list = state_list
        self.stdDeviations = stdDeviations
        self.geom_cnt = geom_cnt
        self.steps = 360//num_ellipse_coords
        self.min_radius = min_radius
        self.epsilon = min_radius/gu.RADIUS_OF_EARTH
        self.clusterCount = 0
        self.clusters = []
        self.featureCount = len(coords)
        self.uniqueFeatureCount = len(np.unique(coords, axis=0))
        
    def choose_most_important_item(self, mask, weights, n = None):
        weights = weights.flatten()
        mask = mask.flatten()
        f, u = pd.factorize(mask)
        sums = dict(zip(u,np.bincount(f, weights)))
        sums.pop(-1, None)
        top_n = sorted(sums, key = sums.get, reverse=True)[:n]
        if n == 1:
            top_n = top_n[0]
        return top_n

    def choose_unique_items(self, items):
        return np.unique(items)

        
    def find_clusters(self, coords):
        cluster_labels = np.ones(self.featureCount)
        min_sample = np.sqrt(self.featureCount)
        try:
            db = DBSCAN(eps = self.epsilon, min_samples=min_sample, 
                        algorithm='ball_tree', 
                        metric='haversine').fit(np.radians(coords))
            cluster_labels = db.labels_
        except Exception as e:
            raise ValueError("DBScan Error " + e)

        cluster_label_set = self.choose_unique_items(cluster_labels)
        if len(cluster_label_set) == 0:
            raise ValueError('Unable to form clusters')
        return cluster_labels, cluster_label_set
    
    def getWeightedCentroid(self, coords, weights):
        weights.shape = self.featureCount, 1
        xyWeighted = weights * coords
        weightSum = weights.sum()
        centers = xyWeighted.sum(0) / weightSum  
        return centers.tolist()
        
    def incrementClusterCount(self):
        self.clusterCount += 1
    
    def return_obj(self):
        clusters = []
        for cluster in self.__dict__["clusters"]:
            clusters.append(cluster.return_obj())
        self.__dict__["clusters"] = clusters
        return self.__dict__
    
   
        
    
    def create_geo(self):
        try:
            
            max_weight_index = np.argmax(self.weights)
            if self.uniqueFeatureCount == 1:
                
                cluster = Cluster(self)
                cluster.city = self.city_list[max_weight_index]
                cluster.state = self.state_list[max_weight_index]
                cluster.createFallbackCluster(self, self.coords, self.weights)

            elif self.uniqueFeatureCount <= 7:
                cluster = Cluster(self)
                try:      
                    cluster.centroid = self.getWeightedCentroid(self.coords, self.weights)

                    max_weighed_coords = self.coords[max_weight_index]
                    gcd_center_maxwtcoord = gu.gcd(cluster.centroid, max_weighed_coords, gu.RADIUS_OF_EARTH)
                    if gcd_center_maxwtcoord > self.min_radius:
                        cluster.centroid = max_weighed_coords
                        cluster.setPositionalIndices(max_weight_index, self.weights[max_weight_index])
                    
                    cluster.city = self.city_list[max_weight_index]
                    cluster.state = self.state_list[max_weight_index]
                    cluster.setEllipseParams(self, radius = self.min_radius, center = cluster.centroid)    
                    cluster.createEllipseCoords(self)
                except Exception as e:
                    cluster.createFallbackCluster(self, self.coords, self.weights )
                    cluster.error = e


            else:
                cluster_labels, cluster_label_set = self.find_clusters(self.coords)
                most_important_clusters = self.choose_most_important_item(cluster_labels, self.weights, self.geom_cnt)
                for cluster_label in most_important_clusters:
                    cluster = Cluster(self)
                    pos_indices = (cluster_labels == cluster_label)
                    xy = self.coords[pos_indices]
                    w = self.weights[pos_indices]
                    cluster_cities = self.city_list[pos_indices]
                    cluster.city = self.choose_most_important_item(cluster_cities, w, 1)
                    cluster_states = self.state_list[pos_indices]
                    cluster.state = self.choose_most_important_item(cluster_states, w, 1)

                    try:
                        numFeatures = len(xy)
                        w.shape = numFeatures, 1
                        cluster.setPositionalIndices(pos_indices.tolist(), w)
                        numUniqueFeatures = len(np.unique(xy, axis = 0))

                        if numUniqueFeatures <= 7: 
                            cluster.setEllipseParams(weights= w,\
                                                     coords = xy,\
                                                     radius = self.min_radius)
                        else:
                            cluster.setEllipseParams(is_ellipse = True, \
                                                     weights = w, \
                                                     coords = xy, \
                                                     stdDeviations = self.stdDeviations)
                            
                        cluster.centroid = cluster.ellipse_params["centroid"].tolist()
                        cluster.createEllipseCoords(self)
                    except Exception as e:
                        cluster.createFallbackCluster(self, xy, w)
                        cluster.error = e
        except Exception as e:
            self.error = e
                
        return self.return_obj()
        

class Cluster():
                           
    def __init__(self, loc):
        self.identifier = "%s_%s".format(loc.identifier, loc.clusterCount)
        self.rank = loc.clusterCount + 1
        loc.incrementClusterCount()
        loc.clusters.append(self)
        self.pos_indices = [True] * loc.featureCount
        self.cluster_weight = np.sum(loc.weights[self.pos_indices])
    
    def setPositionalIndices(self, indices, weights):
        if type(indices) == np.int64:
            new_indices = [False] * len(self.pos_indices)
            new_indices[indices] = True
            self.pos_indices = new_indices
        else:
            self.pos_indices = indices
        self.uniqueFeatureCount = np.sum(self.pos_indices)
        self.cluster_weight = np.sum(weights)
    
    def createFallbackCluster(self, loc, xy, w):
        try:
            max_weight_index = np.argmax(w)
            max_weighed_coords = xy[max_weight_index]
            self.centroid = max_weighed_coords.tolist()
            self.setPositionalIndices(max_weight_index, w[max_weight_index])
            self.setEllipseParams(radius = loc.min_radius, center = self.centroid)
            self.createEllipseCoords(loc)
        except Exception as e:
            self.error = e
    
    def setEllipseParams(self,\
                         is_ellipse = False,\
                         radius = None, \
                         center = None, \
                         coords = None, \
                         weights = None, \
                         stdDeviations = None):
        if not is_ellipse:
            if radius is not None and center is not None:
                self.ellipse_params = \
                {"shape" : "circle", "parameters" : {"distance" : radius}, "centroid" : center}
            elif coords is not None:
                self.ellipse_params = \
                ellipse.CalculateStandardCircleParams(weights, coords, radius)
        else:
            self.ellipse_params = \
            ellipse.CalculateStandardEllipseParams(weights, coords, stdDeviations)
    
    def PolyArea(self):
        c = np.array(self.polygon)
        x = c[:,0]
        y = c[:,1]
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))) 
    
    def createEllipseCoords(self, loc):
        self.polygon = ellipse.GenerateEllipseCoordinates(self.ellipse_params, loc.steps, loc.min_radius)
        self.area = self.PolyArea()
            
        
    def return_obj(self):
        return self.__dict__
