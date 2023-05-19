# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 14:37:00 2021

@author: ABRAHIMI
"""

import pandas as pd
import numpy as np 
import copy


def zeroCoupons(df):

    discFact = df.copy(deep = True)
    dfZC = df.copy(deep = True)    
    
    for i in range(1,len(df.columns))    :
        discFact.iloc[:,i] = ((1-df.iloc[:,i]*(discFact.iloc[:,1:i].sum(axis=1)))/(1+df.iloc[:,i]))  
        dfZC.iloc[:,i] = (discFact.iloc[:,i]**(-1/i)-1)

    return dfZC



def interpo(baseEntree, prefixe='', suffixe ='', seuil = 10):
    #Interpolation des valeurs manquantes
    base = copy.deepcopy(baseEntree)
    colBase = base.columns
    indexBase = base.index
    base.columns =[int(x.replace(prefixe,'').replace(suffixe,'').strip()) for x in base.columns]
    base = base.dropna(thresh = seuil) #au moins 10 valeurs renseignées
    base = base.interpolate(method = "cubic",
                                    limit_direction = "both",
                                    axis = 1)  
    return base


def fusionneColonne(base):
             
    courbe = pd.DataFrame(base.iloc[:,0])
    courbe.columns = ['Date']
    courbe = courbe['Date'].dropna()

    for i in range(0,base.columns.size,2) :
        temp = base.iloc[:,[i,i+1]]
        temp.columns = ['Date', temp.columns[0]]
        temp = temp.dropna()
        courbe = pd.merge(courbe, temp, on='Date', how='outer')
                
        courbe = courbe.sort_values(by='Date')
        courbe.set_index('Date',inplace = True)  
    
    return courbe
    
def centreReduit(df):
    moyET = np.zeros([0,2])
            
    for c in df.columns:
        moyenne = np.mean(df[c])
        ecartType = np.std(df[c])
        df[c]=(df[c]-moyenne)/ecartType
        moyET = np.vstack((moyET, [moyenne, ecartType]))
        print (str(c) + ' moyenne : ' +str(moyenne*100) + ' - écarttype : ' +str(ecartType*100) )
    
    return df, moyET


def filtreDate(df, dateDebut, dateFin):    
    masqueDate = (df.index >= dateDebut) & (df.index <= dateFin)
    df = df[masqueDate].copy(deep = True)

    return df
    