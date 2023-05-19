# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 11:21:49 2023

@author: ABRAHIMI
"""


import pagndas as pd
import numpy as np
import math
import time
import copy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from itertools import product    
import seaborn as sns
from Simulations import *
from ImportetTraitementdesBases import *
from Taux import *
from creerPDF import *

def getFonct(typeModele):
    if typeModele == 'NS' : #Nelson-Siegel
        fonct = lambda τ, λ1, β0, β1, β2 : \
                                                β0 + \
                                                β1 * (1-np.exp(-τ/λ1))/(τ/λ1) + \
                                                β2 * ((1-np.exp(-τ/λ1))/(τ/λ1)-np.exp(-τ/λ1))  
        nParam = 4 
        nλ = 1
        nβ = 3
    if typeModele == 'NSS' : # Nelson-Siegel-Svensson
        fonct = lambda τ, λ1, λ2, β0, β1, β2, β3 : \
                                                β0 + \
                                                β1 * (1-np.exp(-τ/λ1))/(τ/λ1) + \
                                                β2 * ((1-np.exp(-τ/λ1))/(τ/λ1)-np.exp(-τ/λ1)) + \
                                                β3 * ((1-np.exp(-τ/λ2))/(τ/λ2)-np.exp(-τ/λ2))
        nParam = 6
        nλ = 2
        nβ = 4
    if typeModele == 'NSSF' : # Nelson-Siegel-Svensson-Filipovic
        fonct = lambda τ, λ1, λ2, β0, β1, β2, β3 : \
                                                β0 + \
                                                β1 * (1-np.exp(-τ/λ1))/(τ/λ1) + \
                                                β2 * ((1-np.exp(-τ/λ1))/(τ/λ1)-np.exp(-τ/λ1)) + \
                                                β3 * (τ/λ2)**2 * np.exp(-τ/λ2)
        nParam = 6
        nλ = 2        
        nβ = 4
   
    return fonct, nParam, nλ, nβ
        
   
# def appModel(  τ, vLambdas, vBetas):
#     return fonct(listeMaturites[x], *vLambdas, *vBetas)
   

def getCoeff(typeModele, τ, vLambdas):    
   
    λ1 = vLambdas[0]
   
    cβ0 = 1
    cβ1 = (1-np.exp(-τ/λ1))/(τ/λ1)
    cβ2 = (1-np.exp(-τ/λ1))/(τ/λ1)-np.exp(-τ/λ1)

    if typeModele == 'NS' :
        return [cβ0, cβ1, cβ2]
    if typeModele == 'NSS' :
        λ2 = vLambdas[1]
        cβ3_NSS = (1-np.exp(-τ/λ2))/(τ/λ2)-np.exp(-τ/λ2)
        return [cβ0, cβ1, cβ2, cβ3_NSS]
    if typeModele == 'NSSF' :
        λ2 = vLambdas[1]
        cβ3_NSSF = (τ/λ2)**2 * np.exp(-τ/λ2)
        return [cβ0, cβ1, cβ2, cβ3_NSSF]

                             
def erreurQuad( vLambdas, vBetas, pointsCourbe, listeMaturites, fonct):
    erreur = [(pointsCourbe[x]- fonct(listeMaturites[x], *vLambdas, *vBetas))**2 for x in range(len(listeMaturites))]
    return np.sum(erreur)
   
    
def erreurQuadHisto( vLambdas, vBetas, histoCourbe, listeMaturites, fonct, detail = False):
    erreur = np.array([[(histoCourbe[y, x]- fonct(listeMaturites[x], *vLambdas, *vBetas[y,:]))**2 for x in range(len(listeMaturites))] for y in range(len(histoCourbe))])
    if detail : return erreur    
    else : return np.sum(erreur)
                  

def calculBetas(typeModele, pointsCourbe, vLambdas, listeMaturites):
   
    tableCoefs = np.asarray([getCoeff(typeModele, τ, vLambdas) for τ in listeMaturites])
    betas, errQd, *autres  = np.linalg.lstsq(tableCoefs, pointsCourbe, rcond=None)
   
    return betas, errQd


# def calculPoint(point, parametres):
#    return fonct(point, *parametres)


def initialiselambdas(typeModele, bornesLambdas, n, seed):
    s = np.random.RandomState(seed=seed)
   
    if typeModele in ['NSS', 'NSSF'] :
        l1 = s.uniform(bornesLambdas.loc['λ1', 'min'], bornesLambdas.loc['λ1', 'max'],[n,1]).flatten()
        l2 = s.uniform(bornesLambdas.loc['λ2', 'min'], bornesLambdas.loc['λ2', 'max'],[n,1]).flatten()
        return np.vstack([l1, l2]).transpose()  
    else:
        l1 = s.uniform(bornesLambdas.loc['λ1', 'min'], bornesLambdas.loc['λ1', 'max'],[n,1]).flatten()
        return np.vstack([l1]).transpose()


def optGrilleLambdas(typeModele, pointsCourbe, listeMaturites, bornesLambdas, fonct, pas):
    #Construction de la grille
    λ1min = bornesLambdas.loc['λ1', 'min']
    λ1max = bornesLambdas.loc['λ1', 'max']
        
    if typeModele in ['NSS', 'NSSF']:
        λ2min = bornesLambdas.loc['λ2', 'min']
        λ2max = bornesLambdas.loc['λ2', 'max']            
                  
        rangeλ1 = np.arange(λ1min, λ1max, pas)
        rangeλ2 = np.arange(λ2min, λ2max, pas)
        vLambdas = np.array(list(product(rangeλ1, rangeλ2)))
        vLambdas = vLambdas[vLambdas[:,0]<vLambdas[:,1]]
    else :
        vLambdas = [[x] for x in np.arange(λ1min, λ1max, pas)]

    vBetas = np.array([calculBetas(typeModele, pointsCourbe, x , listeMaturites)[0] for x in vLambdas]) 
    vErreurs= np.array([erreurQuad(np.array(vLambdas[i]), vBetas[i], pointsCourbe, listeMaturites, fonct) for i in range(len(vLambdas))])
    indexErrMin = np.argmin(vErreurs)
   
    return np.hstack([vLambdas[indexErrMin], vBetas[indexErrMin], vErreurs[indexErrMin]])



def optimiseLambdas(typeModele, pointsCourbe, listeMaturites, bornesLambdas, fonct, nbEssais = 10 ,seed = 23, method = 'SLSQP', maxiter = 10000):
   
    if typeModele in ['NSS', 'NSSF']:
        bounds = [(bornesLambdas.loc['λ1', 'min'], bornesLambdas.loc['λ1', 'max']),
                  (bornesLambdas.loc['λ2', 'min'], bornesLambdas.loc['λ2', 'max'])]
    else:
        bounds = [(bornesLambdas.loc['λ1', 'min'], bornesLambdas.loc['λ1', 'max'])]
       
    optLambda = lambda l, y, z : minimize(fun = erreurQuad,
                                            x0 = l,
                                            args = (y, z, listeMaturites, fonct),
                                            method = method,
                                            bounds = bounds,
                                            options={'maxiter':maxiter}).x
 
    listeLambdas = initialiselambdas(typeModele, bornesLambdas, nbEssais, seed)
    vBetas = np.array([calculBetas(typeModele, pointsCourbe, x, listeMaturites)[0] for x in listeLambdas])      
    vLambdas = np.array([optLambda(listeLambdas[i], vBetas[i], pointsCourbe) for i in range(nbEssais)])
    vErreurs = np.array([erreurQuad(vLambdas[i], vBetas[i], pointsCourbe, listeMaturites, fonct) for i in range(nbEssais)])
    indexErrMin = np.argmin(vErreurs)
   
    return np.hstack([vLambdas[indexErrMin], vBetas[indexErrMin], vErreurs[indexErrMin]])


def calibrageModele(label, rep, base, typeModele, listeMaturites, bornesLambdas, pas, indexAxe = []):
    
    if len(indexAxe) == 0:
        indexAxe = range(len(base))
    
    fonct, nParam, nλ, nβ = getFonct(typeModele)
        
    resultatGrille = np.empty((len(base), nParam+1))
    resultatGrille = np.array([optimiseLambdas(typeModele, base[i,:], listeMaturites, bornesLambdas, fonct) for i in range(len(base))])
       
    evolutionLambda = pd.DataFrame(resultatGrille[:,0:nλ], index = indexAxe)
    evolutionBetas = pd.DataFrame(resultatGrille[:,nλ:-1], index = indexAxe)
    evolutionErreur = pd.DataFrame(resultatGrille[:,-1], index = indexAxe)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, 
                                        dpi = 100,
                                        figsize = (6,10))
    #fig.subplots_adjust(hspace=0.3)
        
    ax1.plot(evolutionLambda)
    ax1.set_title('Evolution λ')
    ax2.plot(evolutionBetas)
    ax2.set_title('Evolution β')  
    ax3.plot(evolutionErreur)
    ax3.set_title('Evolution Erreurs')  


    fig.tight_layout()
    plt.savefig(rep + label + ' - ' + 'Evolution λ - β - Erreurs'  + '.png')
    resultatGrilleDf = pd.DataFrame(resultatGrille, index = indexAxe)
    return resultatGrilleDf

def calibrageModeleGrille(label, rep, base, typeModele, listeMaturites, bornesLambdas, pas, indexAxe = []):
    
    if len(indexAxe) == 0:
        indexAxe = range(len(base))
    
    fonct, nParam, nλ, nβ = getFonct(typeModele)
 
    resultatGrille = np.empty((len(base), nParam+1))
    resultatGrille = np.array([optGrilleLambdas(typeModele, base[i,:], listeMaturites, bornesLambdas, fonct, pas) for i in range(len(base))])
     
    
    evolutionLambda = pd.DataFrame(resultatGrille[:,0:nλ], index = indexAxe)
    evolutionBetas = pd.DataFrame(resultatGrille[:,nλ:-1], index = indexAxe)
    evolutionErreur = pd.DataFrame(resultatGrille[:,-1], index = indexAxe)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, 
                                        dpi = 100,
                                        figsize = (6,10))
    #fig.subplots_adjust(hspace=0.3)
        
    ax1.plot(evolutionLambda)
    ax1.set_title('Evolution λ')
    ax2.plot(evolutionBetas)
    ax2.set_title('Evolution β')  
    ax3.plot(evolutionErreur)
    ax3.set_title('Evolution Erreurs')  


    fig.tight_layout()
    plt.savefig(rep + label + ' - ' + 'Evolution λ - β - Erreurs'  + '.png')
    resultatGrilleDf = pd.DataFrame(resultatGrille, index = indexAxe)
    return resultatGrilleDf
 
   



def trouveLoi(label, resultatOptim, typeModele, listeDist):
    

    fonct, nParam, nλ, nβ = getFonct(typeModele)
   
    lambdas = resultatOptim[:,:nλ] 
    dlambdas = np.diff(lambdas,axis = 0) 
   
    betas =  resultatOptim[:,nλ:-1] 
    dbetas = np.diff(betas,axis = 0)
 
    # Détermination des lois
    distributionRetenue = []
    listeParams = []
    
    for i in range(nβ+nλ):
        params, name_params, KS_res  = testDist (np.diff(resultatOptim[:,i] ,axis = 0), listeDist, seuil=0.05)   
        distributionRetenue.append(KS_res.iloc[1,:].idxmax())  
        paramLoi={}
        paramLoi['distName'] = KS_res.iloc[1,:].idxmax()
        paramLoi['parametres'] = [*params[paramLoi['distName']]]
        paramLoi['test KS'] = [KS_res.iloc[1,:].max()]
        listeParams.append(paramLoi)
    
    return lambdas, dlambdas, betas, dbetas, listeParams



def trouveLoiBeta(label, resultatOptim, typeModele, listeDist):
    
    fonct, nParam, nλ, nβ = getFonct(typeModele)
   
    betas =  resultatOptim[:,nλ:-1] 
    dbetas = np.diff(betas,axis = 0)

    # Détermination des lois
    distributionRetenue = []
    listeParams = []
    nβ = nParam-nλ 
    for i in range(nβ):
        params, name_params, KS_res  = testDist (dbetas[:,i], listeDist, seuil=0.05)   
        distributionRetenue.append(KS_res.iloc[1,:].idxmax())  
        paramLoi={}
        paramLoi['distName'] = KS_res.iloc[1,:].idxmax()
        paramLoi['parametres'] = [*params[paramLoi['distName']]]
        paramLoi['testKS'] = [KS_res.iloc[1,:].max()]
        listeParams.append(paramLoi)
    
    return betas, dbetas, listeParams
    

def traceAdequation(base, baseOptim, typeModele, listeDates, nL, nC, label, rep, listeMaturites):
    
    fonct, nParam, nλ, nβ = getFonct(typeModele)

    plt.clf()
    fig = plt.figure(figsize = (nL*3, nC*3), dpi = 100)
                          
    for i in range(0, min(len(listeDates), nL* nC)):
        d  = pd.to_datetime(listeDates[i], format='%Y-%m-%d')
        paramOptim =  baseOptim[baseOptim['Date'] == listeDates[i]].values[0][1:-1] 
        courbeOptim =  [fonct(x, *paramOptim) for x in range(1,31,1)]
        
        ax = fig.add_subplot(nL, nC, i+1)
        ax.set_title(listeDates[i])
        plt.plot(listeMaturites, courbeOptim , '-', label='modèle NSSF')
        plt.plot(listeMaturites, base[base.index == d].to_numpy()[0], 'o', label='données courbe')
    plt.tight_layout()

    plt.savefig(rep + label + '.png', dpi = 100)



                  
def optimiseLambdasHisto(typeModele, histoCourbe, listeMaturites, bornesLambdas, fonct, nbEssais = 10 ,seed = 23, method = 'SLSQP', maxiter = 10000):
   
    if typeModele in ['NSS', 'NSSF']:
        bounds = [(bornesLambdas.loc['λ1', 'min'], bornesLambdas.loc['λ1', 'max']),
                  (bornesLambdas.loc['λ2', 'min'], bornesLambdas.loc['λ2', 'max'])]
    else:
        bounds = [(bornesLambdas.loc['λ1', 'min'], bornesLambdas.loc['λ1', 'max'])]
       
    optLambdaHisto = lambda l, y, z : minimize(fun = erreurQuadHisto,
                                            x0 = l,
                                            args = (y, z, listeMaturites, fonct),
                                            method = method,
                                            bounds = bounds,
                                            options={'maxiter':maxiter}).x
 
    listeLambdas = initialiselambdas(typeModele, bornesLambdas, nbEssais, seed)
    vBetas = np.array([[calculBetas(typeModele, histoCourbe[i,:], x, listeMaturites)[0] for x in listeLambdas] for i in range(len(histoCourbe))])   
    vLambdas = np.array([optLambdaHisto(listeLambdas[i], vBetas[:,i,:], histoCourbe) for i in range(nbEssais)])
    vErreursHisto = np.array([erreurQuadHisto(vLambdas[i], vBetas[:,i,:],  histoCourbe, listeMaturites, fonct) for i in range(nbEssais)])

    indexErrMin = np.argmin(vErreursHisto)
   
    return np.hstack([np.repeat([vLambdas[indexErrMin]], len(histoCourbe),axis = 0), vBetas[:,indexErrMin,:]]), vErreursHisto[indexErrMin]



def simulCourbe(simulationParam, courbeRef, nbSimuls, listeMaturites, fonct):

    courbesSimulees = np.empty((nbSimuls, len(listeMaturites)))
    courbesSimulees = np.array([[fonct(x,*simulationParam[i,:]) for x in listeMaturites] for i in range(nbSimuls)])
    chocsSimules = courbesSimulees - courbeRef
    return courbesSimulees, chocsSimules 


def optimiseDfin(typeModele, valIni, base, listeMaturites, bornesLambdas, fonct, seed = 23, method = 'SLSQP', maxiter = 10000):
   
    if typeModele in ['NSS', 'NSSF']:
        bounds = [(bornesLambdas.loc['λ1', 'min'], bornesLambdas.loc['λ1', 'max']),
                  (bornesLambdas.loc['λ2', 'min'], bornesLambdas.loc['λ2', 'max'])]
    else:
        bounds = [(bornesLambdas.loc['λ1', 'min'], bornesLambdas.loc['λ1', 'max'])]
   
    
    # initialisation 
    vLambdas = np.array(valIni[0], ndmin = 2)
    vBetas = np.array(valIni[1], ndmin = 2)
    vErreurs = np.array(erreurQuad(valIni[0], valIni[1], base[0,:], listeMaturites, fonct), ndmin = 2)
    
    
    for i in range(1, len(base)):
        pointsCourbe = base[i,:]
         
        optLambda = lambda l, y, z : minimize(fun = erreurQuad,
                                                x0 = l,
                                                args = (y, z, listeMaturites, fonct),
                                                method = method,
                                                bounds = bounds,
                                                options={'maxiter':maxiter}).x
        
     
        lambdas = np.array(optLambda(vLambdas[i-1,:], vBetas[i-1,:], pointsCourbe), ndmin = 2)    
        betas = np.array(calculBetas(typeModele, pointsCourbe, lambdas[0], listeMaturites)[0])      
        erreurs = np.array(erreurQuad(lambdas, betas, pointsCourbe, listeMaturites, fonct), ndmin = 2)
        
        vLambdas = np.vstack([vLambdas, lambdas])
        vBetas = np.vstack([vBetas, betas])
        vErreurs = np.vstack([vErreurs, erreurs])
        
    resultat = np.hstack([vLambdas, vBetas, vErreurs])
   
    return resultat


# def lanceAnalyse(cheminBases, rep, fichierCalibrage, baseTnDf, baseTiDf, listeMaturites, listeMaturitesReduite):
        
#     # =============================================================================
#     # Création des bases périodiques
#     # =============================================================================
    
#     #Journalier 
#     baseTn = baseTnDf.to_numpy()*100 
#     baseTi = baseTiDf.to_numpy()

#     # Hebdomadaire
#     baseTnWDf = baseTnDf.reset_index().resample('W-Fri', label='right', closed = 'right', on='Date').last()
#     del baseTnWDf['Date']
#     baseTnW = baseTnWDf.to_numpy()
    
#     baseTiWDf = baseTiDf.reset_index().resample('W-Fri', label='right', closed = 'right', on='Date').last()
#     del baseTiWDf['Date']
#     baseTiW = baseTiWDf.to_numpy()
    
#     # Mensuel
#     baseTnMDf = baseTnDf.reset_index().resample('1M', label='right', closed = 'right', on='Date').last()
#     del baseTnMDf['Date']
#     baseTnM = baseTnMDf.to_numpy()
    
#     baseTiMDf = baseTiDf.reset_index().resample('1M', label='right', closed = 'right', on='Date').last()
#     del baseTiMDf['Date']
#     baseTiM = baseTiMDf.to_numpy()


    
#     # =============================================================================
#     # Calibrage fixe
#     # =============================================================================
    
    
#     # Initialisation des variables
    
#     df = pd.read_excel(io = fichierCalibrage, parse_dates = True, skiprows=3, header = None, index_col = 1).iloc[:,1:]
#     nModeles = df.shape[1]
    
#     for k in tqdm(range(nModeles)):
           
#         # Définition des variables
#         label = df.loc["label"].iloc[k]
#         typeModele = df.loc["typeModele"].iloc[k]
#         dateDebut = df.loc["dateDebut"].iloc[k]
#         dateFin = df.loc['dateFin'].iloc[k]
#         periodicite = df.loc["periode"].iloc[k]
#         indicTR = df.loc["indicTR"].iloc[k]
#         nbSimuls = df.loc["nbSimuls"].iloc[k]
#         nbPeriodes = df.loc["nbPeriodes"].iloc[k]
#         methodOptim = df.loc["methodOptim"].iloc[k]
        
#         pas = df.loc["pasGrille"].iloc[k]        
#         listeBases  = eval(df.loc["listeBases"].iloc[k])
#         bornesLambdas = pd.DataFrame(index = ['λ1', 'λ2'], columns = ['min', 'max'])  
#         bornesLambdas.loc['λ1'] = eval(df.loc["bornesλ1"].iloc[k])
#         bornesLambdas.loc['λ2'] = eval(df.loc["bornesλ2"].iloc[k])
#         listeDist = eval(df.loc["listeDist"].iloc[k])
        
        
#         if indicTR : typeTaux = ['Tn', 'Ti']
#         else : typeTaux = ['Tn', 'Ti', 'Tr']
        
#         # Selection de l'historique
#         baseTnFiltre = filtreDate(listeBases[0], dateDebut, dateFin)
#         baseTn = baseTnFiltre.to_numpy()*100 
#         baseTiFiltre = filtreDate(listeBases[1], dateDebut, dateFin)
#         baseTi = baseTiFiltre.to_numpy()
                
#         indexDates =  baseTnFiltre.index
        
#         # Fonctionnelle du modèle
#         fonct, nParam, nλ, nβ = getFonct(typeModele)
         
#         # Optimisation
#         optimTn, erreurTn = optimiseLambdasHisto(typeModele, baseTn, listeMaturites, bornesLambdas, fonct, method = methodOptim)
#         optimTi, erreurTi = optimiseLambdasHisto(typeModele, baseTi, listeMaturites, bornesLambdas, fonct, method = methodOptim)
        
        
#         # Calcul des Erreurs
#         vErreursHistoTn = np.array(erreurQuadHisto(optimTn[0,:nλ], optimTn[:,nλ:],  baseTn, listeMaturites, fonct, detail = True).sum(axis = 1), ndmin=2).T   
#         vErreursHistoTi = np.array(erreurQuadHisto(optimTi[0,:nλ], optimTi[:,nλ:],  baseTi, listeMaturites, fonct, detail = True).sum(axis = 1), ndmin=2).T
    
#         optimTn = np.concatenate([optimTn, vErreursHistoTn], axis =1)  
#         optimTi = np.concatenate([optimTi, vErreursHistoTi], axis =1)  
    
#         # Tracé des distributions et détermination des lois
#         betasN, dBetasN, listeParamsN = trouveLoiBeta(label + ' - ' + typeTaux[0],  
#                                                   optimTn , 
#                                                   typeModele, 
#                                                   listeDist)
        
#         betasI, dBetasI, listeParamsI = trouveLoiBeta(label + ' - ' + typeTaux[1], 
#                                                   optimTi , 
#                                                   typeModele, 
#                                                   listeDist)
    
#         # Détermination des lois
        
#         join_dBetas = np.hstack([dBetasN, dBetasI])
#         join_dParams =  listeParamsN + listeParamsI
    
#         matriceCorrel = np.corrcoef(join_dBetas, rowvar=False)
        
#         # Génération des trajectoires
#         trajectoires = constructionTrajectoires(matriceCorrel, nbSimuls, nbPeriodes, join_dParams)
#         trajectoiresN = trajectoires[:nβ,:]
#         trajectoiresI = trajectoires[nβ:,:]
    
    
#         # Reconstruction des courbes de taux
#         # Taux nominaux 
        
#         baseλTn = np.ones((nbSimuls,1))*optimTn[0,:nλ]
#         baseλTi = np.ones((nbSimuls,1))*optimTi[0,:nλ]
        
        
#         betasSimulesTn = trajectoiresN.T + betasN[-1,:]
#         betasSimulesTi = trajectoiresI.T + betasI[-1,:]
        
#         simulationParamTn = np.hstack([baseλTn, betasSimulesTn]) 
#         simulationParamTi = np.hstack([baseλTi, betasSimulesTi])
    
#         courbesTn, chocsTn = simulCourbe(simulationParamTn,
#                                                         baseTn[-1,:],
#                                                         nbSimuls, 
#                                                         listeMaturites,
#                                                         fonct)
        
#         courbesTi, chocsTi = simulCourbe(simulationParamTi, 
#                                                         baseTi[-1,:], 
#                                                         nbSimuls, 
#                                                         listeMaturites,
#                                                         fonct)
    
    
#         if indicTR :
#             chocsSimulesTn = pd.DataFrame(chocsTn, columns = [*range(1,31)])
#             chocsSimulesTr = pd.DataFrame(chocsTi, columns = [*range(1,31)])
#             chocsSimules = {'Tn' : chocsSimulesTn, 'Tr' : chocsSimulesTr }
            
#         else : 
#             baseTrDf = pd.DataFrame(np.array([[(( 1 + baseTn[x,y]/100) / (1 + baseTi[x,y]/100 )-1)
#                                                       for y in range(len(listeMaturites))] for x in range(len(baseTn))]),
#                                                     columns = [*range(1,31)])*100
            
#             baseTr = baseTrDf.to_numpy()
            
#             chocsSimulesTn = pd.DataFrame(chocsTn, columns = [*range(1,31)])
#             chocsSimulesTi = pd.DataFrame(chocsTi, columns = [*range(1,31)])
            
#             chocsSimulesTr = pd.DataFrame(np.array([[(( 1 + baseTn[-1,y]/100 + chocsTn[x,y]/100) / (1 + baseTi[-1,y]/100 + chocsTi[x,y]/100)-1-baseTr[-1,y]/100)
#                                                       for y in range(len(listeMaturites))] for x in range(nbSimuls)]),
#                                                     columns = [*range(1,31)])*100
            
#             courbesTr = pd.DataFrame(np.array([[ baseTr[-1,y]/100 + chocsSimulesTr.iloc[x,y]/100
#                                                       for y in range(len(listeMaturites))] for x in range(nbSimuls)]),
#                                                     columns = [*range(1,31)])*100
            
#             chocsSimules = {'Tn' : chocsSimulesTn, 'Ti' : chocsSimulesTi, 'Tr' : chocsSimulesTr}
    
    
    
        
       
#         # Sorties Pour Graphique
#         dictResultat = {}
#         dictResultat['Parametres'] = df.iloc[:,k]
#         dictResultat['listeMaturites'] = listeMaturites
#         dictResultat['listeMaturitesReduite'] = listeMaturitesReduite
#         dictResultat['typeTaux'] = typeTaux  
#         dictResultat['nλ'] = nλ 
#         dictResultat['nβ'] = nβ 
#         dictResultat['listeOptim'] = [pd.DataFrame(optimTn, index = indexDates),
#                                       pd.DataFrame(optimTi, index = indexDates)]
    
#         dictResultat['join_dParams'] = join_dParams
#         dictResultat['listeBfp'] = [bfp, bfpTn, bfpTr]
#         dictResultat['matriceCorrel'] = matriceCorrel
#         dictResultat['indicTR'] = indicTR
        
#         if indicTR : 
#             dictResultat['listeBases']=[baseTn, baseTi]
#             dictResultat['listeChocs'] = [chocsTn, chocsSimulesTr.to_numpy()]
#             dictResultat['listeCourbes'] = [courbesTn, courbesTi]
#         else :
#             dictResultat['listeBases']=[baseTn, baseTi, baseTr]
#             dictResultat['listeChocs'] = [chocsTn, chocsTi, chocsSimulesTr.to_numpy()]
#             dictResultat['listeCourbes'] = [courbesTn, courbesTi, courbesTr.to_numpy()]
    
#         fichierSortie = rep + label + '.pdf'
    
    
#         creationPDF(fichierSortie, dictResultat)
        
        
    
    
   


# Fichier de Lancement
        
# Définition des chemins :


