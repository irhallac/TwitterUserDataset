# ibrahim user2vec model testing class

import os
import numpy as np
import json
import sys
print(sys.executable)
from datetime import datetime 
from numpy import dot
from numpy.linalg import norm


class user2vec_test(object):

    def __init__(self, user_vecs, ctg_vecs):        
        self.user_vecs = user_vecs
        self.ctg_vecs = ctg_vecs
        print("user2vec_test class __init__ method successful")
    
    def getUserCategory(self, user_name): #returns category of a user
        return self.user_vecs[user_name]['category']
    
    def getUserAvgVec(self, user_name):
        return self.user_vecs[user_name]['avr_vec']
        #returns avg vec of a user
    
    def getCtgAvgVec(self, ctg):
        return self.ctg_vecs[ctg]['avr_cat_vec']
        #returns avg vec of a category
        

    def user_relation(self, _x1,_x2,_y1, exclude_list): #x1:paris x2:france y1:london return y2; hopefully england
        x1 = self.getUserAvgVec(_x1)
        x2 = self.getUserAvgVec(_x2)
        y1 = self.getUserAvgVec(_y1)
        x_temp  = np.array(x1) + np.array(x2) - np.array(y1)
        y2 = self.most_similar_user_by_vec(x_temp, exclude_list= exclude_list)
        return y2
      
    #similarity measurement with Euclidean Distance
    def l2(self, a, b):
        # same as np.sqrt(np.sum((x-y)**2))
        return norm(np.array(a)-np.array(b))

    def readJson(self, path):
        with open(path, 'r') as outfile:
            data = json.load(outfile)
        outfile.close()
        return data
    
            
    def most_similar_group(self, _x1): #_x1:username (joe biden - politics ?)
        x1 = self.getUserAvgVec(_x1)
        i = 0
        for ctg in self.ctg_vecs.keys():
            dist = self.l2(x1, self.ctg_vecs[ctg]["avr_cat_vec"])
            if i == 0:
                #print(unm, dist)
                mind = dist
                ust = ctg
            else:
                if dist < mind:
                    mind = dist
                    #print(unm, dist)
                    ust = ctg
            i = i + 1
        return ust
    
    def most_similar_user(self, _x1): #_x1:username who is most similar to x1 ? (joe biden - kamala harris?)
        x1 = self.user_vecs[_x1]["avr_vec"]
        i = 0
        for unm in self.user_vecs.keys():
            if unm != _x1:
                dist = self.l2(x1, self.user_vecs[unm]["avr_vec"])
                if i == 0:
                    #print(unm, dist)
                    mind = dist
                    ust = unm
                else:
                    if dist < mind:
                        mind = dist
                        #print(unm, dist)
                        ust = unm
                i = i + 1
        return ust
        
    def most_similar_user_by_vec(self, _x1, exclude_list): #find closest user for vector _x1
        i = 0
        for unm in self.user_vecs.keys():
            if unm not in exclude_list:
                dist = self.l2(_x1, self.getUserAvgVec(unm))
                if i == 0:
                    mind = dist
                    ust = unm
                else:
                    if dist < mind:
                        mind = dist
                        #print(unm, dist)
                        ust = unm

                i = i + 1
        return ust