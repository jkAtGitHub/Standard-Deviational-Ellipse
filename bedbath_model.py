#!/usr/bin/env python

#===========================================================================
# Algorithm
#    beds | Views              beds | Views
#    -------------             -----------
#      2  | 6                    4  | 7           7/22  (32%)
#      3  | 5         -->        2  | 6    -->    13/22 (60%)
#      4  | 7   (sort on Views)  3  | 5           18/22 (82%)
#      5  | 4                    5  | 4           100%
#    ------------               ------------
#  Take the value that is closer to 80% cumulative percent (4,2,3)
#  Sort the list: (2, 3, 4) min_beds: 2 max_beds: 4
#==========================================================================

#===========================================================================
# Constants local to this file
REQUIRED_CUM_PERCENT_VALUE = 80
#===========================================================================


class CustomObj(object):

    def __init__(self, identifier, count):
        self.identifier = identifier
        self.count = count
        self.cumulativePercent = 0
        
    def getCount(self):
        return self.count
    
    def getCumulativeScore(self):
        return self.cumulativePercent
    
    def getIdentifier(self):
        return self.identifier
        
    def incrCount(self):
        self.count += 1
        
    def setCumulativePercent(self, value):
        self.cumulativePercent = value
        
    def __str__(self):
        return("(identifier:" + str(self.identifier) + " count:" + str(self.count) \
               + " cumulativePercent:" + str(self.cumulativePercent) + ")")

        
def printList(lst):
    print [str(x) for x in lst]


def contructObjectList(listIn):
    objectsDict = {}
    objectsList = list()
    for item in listIn:
        if item is None:
            continue
        customObj = objectsDict.get(item, None)
        if customObj is None:
            customObj = CustomObj(item, 0)
            objectsList.append(customObj)
            objectsDict[item] = customObj
        customObj.incrCount()
    
    return objectsList

    
def create_recommendation(listIn):
    if not listIn:
        raise Exception('Null list passed into the bedbath model')
    arryLength = len(listIn)
    objectsList = contructObjectList(listIn)
        
    objectsList = sorted(objectsList, key=lambda x: x.count, reverse=True)
    cumulativeSum = 0
    for customObj in objectsList:
        cumulativeSum += customObj.getCount()
        customObj.setCumulativePercent((cumulativeSum * 100) / arryLength)
        
    closestValue = min([abs(x.getCumulativeScore() - REQUIRED_CUM_PERCENT_VALUE) for x in objectsList])
    matchedIndex = [i for i, v in enumerate(objectsList) \
                        if abs(v.getCumulativeScore() - REQUIRED_CUM_PERCENT_VALUE) == closestValue]
    if not matchedIndex:
        raise Exception('Unable to find required cumulative percent value')

    objectsList = sorted(objectsList[:matchedIndex[0] + 1], key=lambda x: x.identifier)

    minMaxList = list()
    if (matchedIndex == 0):
        minMaxList.append(objectsList[0].getIdentifier())
        minMaxList.append(objectsList[0].getIdentifier())
    else:
        minMaxList.append(objectsList[0].getIdentifier())
        minMaxList.append(objectsList[matchedIndex[0]].getIdentifier())
    return minMaxList
    
#===========================================================================
