# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:45:38 2023

@author: ABRAHIMI
"""

import scipy.stats as st
import pandas as pd
import numpy as np
import math


def testDist (serie, listeDist, seuil=0.05):
    #Test de Kolmogorov smirnov et retour des paramètre des distributions
      #Définition des paramètres en sortie
      import scipy.stats as st
      params = {}
      name_params ={}
      KS_res = pd.DataFrame()
 
      for dist_name in listeDist:
            dist = getattr(st, dist_name)
            param = dist.fit(serie)
            params[dist_name] = param
            name_params[dist_name] = list(filter(None,[dist.shapes,'loc','scale']))

            # Test de Kolmogorov-Smirnov test
            temp = st.kstest(serie, dist_name, args=param)
            KS_res[dist_name] = [temp.statistic, temp.pvalue]

      return params, name_params, KS_res   
  
    
def ppfInterp(aleas, distribution, plage, loc, scale, *arg):
    from scipy.interpolate import UnivariateSpline #interp1d

    vals = distribution.ppf(plage, loc=loc, scale=scale, *arg)
    ppf = UnivariateSpline(plage, vals, s=10e-10)

    return ppf(aleas)


def simulationsDist(matriceCorrel, nbSimuls, nbPeriodes, listeParams, plage = np.linspace(0.001,0.99,10**3), graine = 1):

     # Décomposition de Cholesky
    C = np.linalg.cholesky(matriceCorrel)
    
    # Génération d'aléas gaussiens
    np.random.seed(graine)

    aleasGauss = np.dot(C,np.random.normal(size = (len(listeParams),nbSimuls*nbPeriodes)))      

    # Construction de la distribution
    aleas = np.empty((len(listeParams),nbSimuls*nbPeriodes))
    i = 0
    
    for vdist in listeParams:
        distribution = getattr(st, vdist['distName'])
          
        print(str(i)+ " - "+ vdist['distName'])
        # Séparation des paramètres
        arg = vdist['parametres'][:-2]
        loc = vdist['parametres'][-2]
        scale = vdist['parametres'][-1]
        
        # Génération des aléas 
        aleasCdf = st.norm.cdf(aleasGauss[i,:])
                  
        aleas[i,:] = ppfInterp(aleasCdf, distribution, plage, loc, scale, *arg) 
        i=i+1

    return aleas


def constructionTrajectoires(matriceCorrel, nbSimuls, nbPeriodes, listeParams):
    simulations =  np.empty((len(listeParams),nbSimuls, nbPeriodes))
    trajectoires = np.empty((len(listeParams),nbSimuls))
    simulationlois = simulationsDist(matriceCorrel, nbSimuls, nbPeriodes, listeParams)
    
    #redimensionement
    for i in range(len(listeParams)):
        simulations[i,:,:] = np.reshape(simulationlois[i,:],(nbSimuls, nbPeriodes))
        trajectoires[i,:] = simulations[i].sum(axis = 1)

    return trajectoires


