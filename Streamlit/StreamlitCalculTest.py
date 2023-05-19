# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:19:40 2023

@author: samym
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from Simulations import *
from Taux import *
from creerPDFv9 import *
from FonctionsNSSFv9 import *
import streamlit


# Partie Import des Données


streamlit.title("Test d'application pour le calcul de chocs de taux en fonction de la paramètrisation d'un fichier Excel")



def lanceAnalyse(rep, fichierCalibrage, baseTnDf, baseTiDf, listeMaturites, listeMaturitesReduite):
        
    # =============================================================================
    # Création des bases périodiques
    # =============================================================================
    
    #Journalier 
    baseTn = baseTnDf.to_numpy()*100 
    baseTi = baseTiDf.to_numpy()

    # Hebdomadaire
    baseTnWDf = baseTnDf.reset_index().resample('W-Fri', label='right', closed = 'right', on='Date').last()
    #del baseTnWDf['Date']
    baseTnW = baseTnWDf.to_numpy()
    
    baseTiWDf = baseTiDf.reset_index().resample('W-Fri', label='right', closed = 'right', on='Date').last()
    #del baseTiWDf['Date']
    baseTiW = baseTiWDf.to_numpy()
    
    #Mensuel
    baseTnMDf = baseTnDf.reset_index().resample('1M', label='right', closed = 'right', on='Date').last()
    #del baseTnMDf['Date']
    baseTnM = baseTnMDf.to_numpy()
    
    baseTiMDf = baseTiDf.reset_index().resample('1M', label='right', closed = 'right', on='Date').last()
    #del baseTiMDf['Date']
    baseTiM = baseTiMDf.to_numpy()


    
    # =============================================================================
    # Calibrage fixe
    # =============================================================================
    
    
    # Initialisation des variables
    
    df = pd.read_excel(io = fichierCalibrage, parse_dates = True, skiprows=3, header = None, index_col = 1).iloc[:,1:]
    nModeles = df.shape[1]
    
    for k in tqdm(range(nModeles)):
           
        # Définition des variables
        label = df.loc["label"].iloc[k]
        typeModele = df.loc["typeModele"].iloc[k]
        dateDebut = df.loc["dateDebut"].iloc[k]
        dateFin = df.loc['dateFin'].iloc[k]
        periodicite = df.loc["periode"].iloc[k]
        indicTR = df.loc["indicTR"].iloc[k]
        nbSimuls = df.loc["nbSimuls"].iloc[k]
        nbPeriodes = df.loc["nbPeriodes"].iloc[k]
        methodOptim = df.loc["methodOptim"].iloc[k]
        
        pas = df.loc["pasGrille"].iloc[k]        
        listeBases  = eval(df.loc["listeBases"].iloc[k])
        bornesLambdas = pd.DataFrame(index = ['λ1', 'λ2'], columns = ['min', 'max'])  
        bornesLambdas.loc['λ1'] = eval(df.loc["bornesλ1"].iloc[k])
        bornesLambdas.loc['λ2'] = eval(df.loc["bornesλ2"].iloc[k])
        listeDist = eval(df.loc["listeDist"].iloc[k])
        
        
        if indicTR : typeTaux = ['Tn', 'Ti']
        else : typeTaux = ['Tn', 'Ti', 'Tr']
        
        # Selection de l'historique
        baseTnFiltre = filtreDate(listeBases[0], dateDebut, dateFin)
        baseTn = baseTnFiltre.to_numpy()*100 
        baseTiFiltre = filtreDate(listeBases[1], dateDebut, dateFin)
        baseTi = baseTiFiltre.to_numpy()
                
        indexDates =  baseTnFiltre.index
        
        # Fonctionnelle du modèle
        fonct, nParam, nλ, nβ = getFonct(typeModele)
         
        # Optimisation
        optimTn, erreurTn = optimiseLambdasHisto(typeModele, baseTn, listeMaturites, bornesLambdas, fonct, method = methodOptim)
        optimTi, erreurTi = optimiseLambdasHisto(typeModele, baseTi, listeMaturites, bornesLambdas, fonct, method = methodOptim)
        
        
        # Calcul des Erreurs
        vErreursHistoTn = np.array(erreurQuadHisto(optimTn[0,:nλ], optimTn[:,nλ:],  baseTn, listeMaturites, fonct, detail = True).sum(axis = 1), ndmin=2).T   
        vErreursHistoTi = np.array(erreurQuadHisto(optimTi[0,:nλ], optimTi[:,nλ:],  baseTi, listeMaturites, fonct, detail = True).sum(axis = 1), ndmin=2).T
    
        optimTn = np.concatenate([optimTn, vErreursHistoTn], axis =1)  
        optimTi = np.concatenate([optimTi, vErreursHistoTi], axis =1)  
    
        # Tracé des distributions et détermination des lois
        betasN, dBetasN, listeParamsN = trouveLoiBeta(label + ' - ' + typeTaux[0],  
                                                  optimTn , 
                                                  typeModele, 
                                                  listeDist)
        
        betasI, dBetasI, listeParamsI = trouveLoiBeta(label + ' - ' + typeTaux[1], 
                                                  optimTi , 
                                                  typeModele, 
                                                  listeDist)
    
        # Détermination des lois
        
        join_dBetas = np.hstack([dBetasN, dBetasI])
        join_dParams =  listeParamsN + listeParamsI
    
        matriceCorrel = np.corrcoef(join_dBetas, rowvar=False)
        
        # Génération des trajectoires
        trajectoires = constructionTrajectoires(matriceCorrel, nbSimuls, nbPeriodes, join_dParams)
        trajectoiresN = trajectoires[:nβ,:]
        trajectoiresI = trajectoires[nβ:,:]
    
    
        # Reconstruction des courbes de taux
        # Taux nominaux 
        
        baseλTn = np.ones((nbSimuls,1))*optimTn[0,:nλ]
        baseλTi = np.ones((nbSimuls,1))*optimTi[0,:nλ]
        
        
        betasSimulesTn = trajectoiresN.T + betasN[-1,:]
        betasSimulesTi = trajectoiresI.T + betasI[-1,:]
        
        simulationParamTn = np.hstack([baseλTn, betasSimulesTn]) 
        simulationParamTi = np.hstack([baseλTi, betasSimulesTi])
    
        courbesTn, chocsTn = simulCourbe(simulationParamTn,
                                                        baseTn[-1,:],
                                                        nbSimuls, 
                                                        listeMaturites,
                                                        fonct)
        
        courbesTi, chocsTi = simulCourbe(simulationParamTi, 
                                                        baseTi[-1,:], 
                                                        nbSimuls, 
                                                        listeMaturites,
                                                        fonct)
    
    
        if indicTR :
            chocsSimulesTn = pd.DataFrame(chocsTn, columns = [*range(1,31)])
            chocsSimulesTr = pd.DataFrame(chocsTi, columns = [*range(1,31)])
            chocsSimules = {'Tn' : chocsSimulesTn, 'Tr' : chocsSimulesTr }
            
        else : 
            baseTrDf = pd.DataFrame(np.array([[(( 1 + baseTn[x,y]/100) / (1 + baseTi[x,y]/100 )-1)
                                                      for y in range(len(listeMaturites))] for x in range(len(baseTn))]),
                                                    columns = [*range(1,31)])*100
            
            baseTr = baseTrDf.to_numpy()
            
            chocsSimulesTn = pd.DataFrame(chocsTn, columns = [*range(1,31)])
            chocsSimulesTi = pd.DataFrame(chocsTi, columns = [*range(1,31)])
            
            chocsSimulesTr = pd.DataFrame(np.array([[(( 1 + baseTn[-1,y]/100 + chocsTn[x,y]/100) / (1 + baseTi[-1,y]/100 + chocsTi[x,y]/100)-1-baseTr[-1,y]/100)
                                                      for y in range(len(listeMaturites))] for x in range(nbSimuls)]),
                                                    columns = [*range(1,31)])*100
            
            courbesTr = pd.DataFrame(np.array([[ baseTr[-1,y]/100 + chocsSimulesTr.iloc[x,y]/100
                                                      for y in range(len(listeMaturites))] for x in range(nbSimuls)]),
                                                    columns = [*range(1,31)])*100
            
            chocsSimules = {'Tn' : chocsSimulesTn, 'Ti' : chocsSimulesTi, 'Tr' : chocsSimulesTr}
    
    
    
        
       
        # Sorties Pour Graphique
        dictResultat = {}
        dictResultat['Parametres'] = df.iloc[:,k]
        dictResultat['listeMaturites'] = listeMaturites
        dictResultat['listeMaturitesReduite'] = listeMaturitesReduite
        dictResultat['typeTaux'] = typeTaux  
        dictResultat['nλ'] = nλ 
        dictResultat['nβ'] = nβ 
        dictResultat['listeOptim'] = [pd.DataFrame(optimTn, index = indexDates),
                                      pd.DataFrame(optimTi, index = indexDates)]
    
        dictResultat['join_dParams'] = join_dParams
        dictResultat['matriceCorrel'] = matriceCorrel
        dictResultat['indicTR'] = indicTR
        
        if indicTR : 
            dictResultat['listeBases']=[baseTn, baseTi]
            dictResultat['listeChocs'] = [chocsTn, chocsSimulesTr.to_numpy()]
            dictResultat['listeCourbes'] = [courbesTn, courbesTi]
        else :
            dictResultat['listeBases']=[baseTn, baseTi, baseTr]
            dictResultat['listeChocs'] = [chocsTn, chocsTi, chocsSimulesTr.to_numpy()]
            dictResultat['listeCourbes'] = [courbesTn, courbesTi, courbesTr.to_numpy()]
    
        fichierSortie = rep + label + '.pdf'
    
    
        creationPDF(fichierSortie, dictResultat)
        
        
Fichier=streamlit.file_uploader("Choississez ici le fichier de Courbe de Taux puis le Fichier de Paramètrage de Calibration",accept_multiple_files=True)


try:
    if len(Fichier)==2:
        CourbeTaux=Fichier[0]
        fichierCalibrage=Fichier[1]
        baseNominaux=pd.read_excel(CourbeTaux,sheet_name='Nominal',parse_dates=True,skiprows=3)
        baseInflation=pd.read_excel(CourbeTaux,sheet_name='Inflation',parse_dates=True,skiprows=3)
        df=pd.read_excel(fichierCalibrage,parse_dates=True,skiprows=3,header=None,index_col=1).iloc[:,1:]
        baseNominaux = fusionneColonne(baseNominaux).filter(regex=('EUSA'+".*"))
        baseInflation = fusionneColonne(baseInflation).filter(regex=('FRSWI'+".*"))
        listeSuppNom = ['EUSA'+ str(x) +' Curncy' for x in [11,13,14,16,17,18,19,21,22,24,26,27,28,29]]
        baseNominaux[listeSuppNom] = baseNominaux[listeSuppNom].drop(baseNominaux[listeSuppNom].index)
        listeSuppInf = ['FRSWI'+ str(x) +' Curncy' for x in [11,13,14,16,17,18,19,21,22,24,26,27,28,29]]
        baseInflation[listeSuppInf] = baseInflation[listeSuppInf].drop(baseInflation[listeSuppInf].index)
        baseNominauxInterp= interpo(baseNominaux, prefixe = 'EUSA', suffixe = 'Curncy').dropna()
        baseInflationInterp= interpo(baseInflation, prefixe = 'FRSWI', suffixe = 'Curncy').dropna()
        streamlit.subheader("Voici la base de Travail des Taux Nominaux")
        streamlit.write(baseNominauxInterp)
        streamlit.subheader("Voici le Fichier de Paramétrage que vous avez choisi")
        streamlit.write(df)
        listeMaturites=[*range(1,31)]
        listeMaturitesReduite=[1,10,30]
        baseTnDf = baseNominauxInterp.add_prefix('Tn')/100
        baseTnDf = zeroCoupons(baseTnDf)#découponnage Base Tn
        baseTiDf = baseInflationInterp.add_prefix('Ti')
        #Fusion puis séparation
        baseTnTi = pd.merge(baseTnDf, baseTiDf, on='Date', how = 'inner')
        baseTnDf = baseTnTi.iloc[:,:30]
        baseTn = baseTnDf.to_numpy()*100 
        baseTiDf = baseTnTi.iloc[:,30:]
        baseTi = baseTiDf.to_numpy()
        rep=streamlit.text_input("Ecrivez le répertoire vers lequel vous souhaitez que l'analyse sous format PDF soit stocké. N'oubliez pas d'ajouter un \ pour que le fichier soit bien sauvegardé dans le dernier dossier écrit","Inscrivez un répertoire valide ")
        cheminSorties=rep
        if streamlit.button("Si vous cliquez sur ce bouton, l'analyse sera lancée"):
            lanceAnalyse(rep,fichierCalibrage,baseTnDf,baseTiDf,listeMaturites,listeMaturitesReduite)
    elif len(Fichier)!=2:
        raise TypeError("Importez seulement les 2 fichiers Excel dans le bon ordre")
except ValueError:
    print("Vous n'avez pas importé les 2 fichiers : Courbe de Taux puis Paramètres de Calibration")