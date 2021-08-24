# user2vec class for testing
# paper version_1.0

import os
#import pandas as pd
import numpy as np
import json
import sys
print(sys.executable)
from datetime import datetime 
from numpy import dot
from numpy.linalg import norm


# ibrahim user2vec model testing class
class user2vec_test(object):

    def __init__(self, user_vecs, ctg_vecs):        
        self.user_vecs = user_vecs
        self.ctg_vecs = ctg_vecs
        #print("user2vec_test class __init__ method successful")
        self.acc = 0
        

    
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
    
    def same_cat(self, ctg, topN):
        count = 0
        for i in topN:  #each i is a tuple
            if ctg == self.getUserCategory(i[0]):
                count += 1
        return count    

    def l2(self, a, b):
        # same as np.sqrt(np.sum((x-y)**2))
        return norm(np.array(a)-np.array(b))

    def readJson(self, path):
        with open(path, 'r') as outfile:
            data = json.load(outfile)
        outfile.close()
        return data
    
    def list_categories(self):
        print("list_categories")
        for ctg in self.ctg_vecs.keys():
            print(ctg,"-", end =" ")
            
    def list_group_similarity(self, _x1): #x1: list group distances for x1
        x1 = self.user_vecs[_x1]["avr_vec"]
        i = 0
        for ctg in self.ctg_vecs.keys():
            dist = self.l2(x1, self.getCtgAvgVec(ctg))
            print(ctg, " %.2f"%(dist))
            
    def most_similar_group(self, _x1): #x1:billgates who is most similar to x1 ??

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
    
    def most_similar_user(self, _x1): #x1:billgates who is most similar to x1 ??

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
    
##### .   .    first acc approach   .  .   #####
    
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

    
    def calc_accuracy(self):
        # overall acc. how many users are closest to their group?
        # 1
        #print("overall acc. how many users are closest to their group?")
        true_p = 0
        total_p = 0
        for unm in self.user_vecs.keys():
            total_p += 1
            if self.most_similar_group(unm) == self.getUserCategory(unm):
                true_p += 1
        #msg = "Total Acc is", true_p,"/",total_p, " = ", true_p/total_p
        msg = "General Acc is " + str(true_p) + "/" + str(total_p) + " = " + str(true_p/total_p) + "\n"
        self.acc = true_p/total_p
        return msg
        
    def calc_accuracy_by_group(self):
        # overall acc. how many users are closest to their group?
        # 1
        msg = ""
        msg += "overall acc. how many users are closest to their group \n "
        for ctg in self.ctg_vecs.keys():
            true_p = 0
            total_p = 0
            for unm in self.user_vecs.keys():
                if self.getUserCategory(unm) == ctg:
                    total_p += 1
                    if self.most_similar_group(unm) == self.getUserCategory(unm):
                        true_p += 1
            msg += ctg + " " + str(true_p) + "/" + str(total_p) + " Acc " + str(true_p/total_p) + "\n"

        msg += "\n"
        return msg
            
    def group_acc1(self, nums):
        print("get N closest users to a group. how many are in that group ? \n")
        for N in nums:
            avg_acc = 0
            for ctg in self.ctg_vecs.keys():
                tt = {}
                avg_cat = self.getCtgAvgVec(ctg)
                for unm in self.user_vecs.keys():
                    tt[unm] = self.l2(avg_cat, self.getUserAvgVec(unm))

                topN = sorted(tt.items(), key=lambda x:-x[1])[-N:]
                acc = self.same_cat(ctg, topN)/N
                avg_acc += acc
                print("top-"+str(N),ctg, " %.2f"%(acc))
            avg_acc = avg_acc/5
            print("avg_acc =", avg_acc) 
            print("-"*20)

##### .   .    second acc approach   .  .   #####
    def group_user_dist_dict(self):
        group2user = {}
        for ctg in self.ctg_vecs.keys():
            avg_cat = self.getCtgAvgVec(ctg) 
            group2user[ctg] = {}
            for unm in [unm for unm in self.user_vecs.keys() if self.getUserCategory(unm) == ctg]:
                group2user[ctg][unm] = self.l2(avg_cat, self.getUserAvgVec(unm))
        return group2user
    
    def selectTestUsers(self, num_users):
    # her gruba ait num_users tane en yakın kullanıcıyı seç
    # grubu en iyi temsil eden num_users X num_of_groups return adet kullanıcı
        testUsers = []
        for ctg in self.ctg_vecs.keys():
            N = num_users
            topN = sorted(self.group2user[ctg].items(), key=lambda x:-x[1])[-N:]
            testUsers.extend([x[0] for x in topN])
        return testUsers
    
    def re_flat(self, user_mat, indice):
        ux, uy = user_mat.shape
        i = int(indice/ux)
        j = indice - i*ux
        return i,j

    #re_flat(user_mat, 19)

    def min_values(self, user_mat, n):
        # find n min values of the user_mat
        srt = np.argsort(user_mat.flat)[:n]

        # return list of i,j 
        return [(self.re_flat(user_mat, i)) for i in srt]

    # matristeki en yakın kişilerin aynı grupta olup olmadığını göster
    def userVecScore(self, testuser, users):    
        usr1 = testuser[users[0]]
        usr2 = testuser[users[1]]
        return self.getUserCategory(usr1) == self.getUserCategory(usr2)
    #userVecScore(testUsers, (9, 10))
    
    def most_similar_acc2(self, num_users):
        
        print("How many of closest different users in ", num_users*5,"X",num_users*5, "matrix -")
        print("-belong to same group " )

        # test edilecek kişilerin birbirine benzerlik matrisini oluştur
        testUsers = self.selectTestUsers(num_users)
        
        high_value = 99
        n = len(set(testUsers))
        user_mat = np.zeros((n, n))

        for i in range(n):
            #vec1 = user_vecs[testUsers[i]]["avr_vec"]
            vec1 = self.getUserAvgVec(testUsers[i])

            for j in range(n):
                if i >= j:
                    user_mat[i,j] = high_value
                elif i < j:
                    user_mat[i,j] = self.l2(vec1, self.getUserAvgVec(testUsers[j]))
                    
        # if testUsers = 10 than there can be 
        # maximum 45 meaningful distance comparisons
        # 5 of them should be in the same group
        
        temp_true = int(5*num_users/2)
        #print(temp_true, "(temp_true) adet ikili bulunmakta")
        most_similar = self.min_values(user_mat, temp_true)
        #print(len(most_similar), " len(most_similar)")

        sim_scores = [self.userVecScore(testUsers, t) for t in most_similar]

        sim_acc = (sim_scores.count(True)/len(sim_scores))*100
        print(sim_acc)
        #print(sim_scores)
        print("\n")
        
        
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# twcollector1: Economy and Finance -- blue -- darkblue
# twcollector2: crypto -- darkred -- maroon
# twcollector3: Technology -- springgreen -- darkgreen
# twcollector4: Fashion and lifestyle -- magenta  -- darkmagenta
# twcollector5: Politics -- orange -- yellow

#colors = {'1':'blue', '1_1':'darkblue', '2':'peru', '2_2':'sienna', 
#          '3':'springgreen', '3_3':'darkgreen','4':'magenta', '4_4':'darkmagenta', 
#          '5':'orange', '5_5':'yellow'}


def plot_model(uvt):
    
    colors = {'1':'blue', '1_1':'darkblue', '2':'darkred', '2_2':'maroon', 
          '3':'springgreen', '3_3':'darkgreen','4':'magenta', '4_4':'darkmagenta', 
          '5':'orange', '5_5':'yellow'}
    
    arr_x = [] 
    arr_y = [] 
    arr_c = [] 


    for unm in uvt.user_vecs.keys():
        _c = uvt.user_vecs[unm]["category"][-1]
        arr_c.append(colors[_c])
        arr_y.append(unm+"_"+_c)
        arr_x.append(uvt.user_vecs[unm]["avr_vec"])

    print('%s adet vektör bulundu.' % len(arr_y))
    
    for ctg in uvt.ctg_vecs.keys():
        _c = ctg[-1]
        arr_c.append(colors[_c + "_" + _c])
        arr_y.append(ctg)
        arr_x.append(uvt.ctg_vecs[ctg]["avr_cat_vec"])
    
    arr_s = [10 for n in range(len(arr_y))]
    for i in range(1,6):
        arr_s[-1*i] = arr_s[-1*i]*10
        

    # önce bir TSNE modeli nesnesi oluşturulur
    model = TSNE(learning_rate = 200)
    transformed = model.fit_transform(arr_x)
    
    xs = transformed[:,0]
    ys = transformed[:,1]
    
    plt.scatter(xs, ys, s=arr_s, c=arr_c)