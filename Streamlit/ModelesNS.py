# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:00:33 2023

@author: smekkaoui
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class ModelesNelsonSiegel():
    
    def __init__(self,b0,b1,b2,lambda1):
        self.b0=b0
        self.b1=b1
        self.b2=b2
        self.lambda1=lambda1
        
    def Fonctionnelle(self,t):
        pass
    
    
    def CalculTaux(self,T):
        Durée=np.arange(0,T+1/12,1/12)
        L=[]
        for element in Durée :
            L.append(self.Fonctionnelle(element))
        dictN = {'Temps' : Durée, 'Taux' : L}
        TauxN = pd.DataFrame.from_dict(data = dictN)
        return TauxN

    def DiffusionTaux(self,T):  
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        A=self.CalculTaux(T)
        ax.plot(A["Temps"],A["Taux"])
        ax.set_ylabel("Taux d'intéret")
        ax.set_title("Diffusion des courbes de taux")
        ax.set_xlabel("Temps en années")
        return fig 
        
    





