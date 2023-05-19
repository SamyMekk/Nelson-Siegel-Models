# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:51:27 2023

@author: abrahimi
"""


import seaborn as sns
import pandas as pd
import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d  # Fonction pour la 3D
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.artist import allow_rasterization

def formatTable(ax, table):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_facecolor("white")


    header_color='#40466e'
    row_colors=['#f1f1f2', 'w']
    edge_color='w'
    header_columns=0
     
    tableau = ax.table(table.values,
                       colLabels = table.columns,
                       loc = 'center')
    
    for m, n in tableau.get_celld():
        cellule = tableau[m, n]
        cellule.set_height(0.1)
        cellule.set_edgecolor(edge_color)
        if m == 0 or m < header_columns:
            cellule.set_text_props(weight='bold', color='w')
            cellule.set_facecolor(header_color)
        else:
            cellule.set_facecolor(row_colors[m%len(row_colors) ])   
    
    return ax


def creationPDF(fichierSortie, dictResultat):
    
    from matplotlib import cm
    mpl.rcParams.update(mpl.rcParamsDefault)
   
    #récupération des données
    
    parametres = dictResultat['Parametres']
    listeMaturites = dictResultat['listeMaturites']
    listeMaturitesReduite = dictResultat['listeMaturitesReduite'] 
    typeTaux= dictResultat['typeTaux'] 
    nλ = dictResultat['nλ'] 
    nβ = dictResultat['nβ'] 
    indicTR = dictResultat['indicTR'] 
    listeBases = dictResultat['listeBases']
    listeOptim = dictResultat['listeOptim'] 
    listeCourbes = dictResultat['listeCourbes']
    listeChocs = dictResultat['listeChocs'] 
    join_dParams =  dictResultat['join_dParams']  
    matriceCorrel = dictResultat['matriceCorrel']
    legendλ = ['λ1', 'λ2']
    legendβ = ['β0', 'β1', 'β2', 'β3']
    
    pp = PdfPages(fichierSortie)

    
    # Page 0 : Paramètres
    #--------------------------------------------  
    

    custom_params = {"axes.spines.right": False, "axes.spines.top": False,
                     "axes.spines.left": False, "axes.spines.bottom": False}
    sns.set_theme(style="white", rc=custom_params)  
    fig, axis = plt.subplots(1, 1, dpi = 150, figsize = (13,6))
      
    tabParam = parametres.reset_index().iloc[1:,:]
    tabParam.columns = ['Paramètres', parametres.iloc[0]]
    formatTable(axis,  tabParam)
    
    fig.tight_layout()
    plt.savefig(pp, format='pdf', dpi = 100)
    plt.close('all')
    
    # Page 1 : Evolution des λ, β, Erreurs
    #--------------------------------------------
    sns.set_theme()
    fig, axis = plt.subplots(3, 2, dpi = 100, figsize = (10,6))
    
    for k in range(2):

        axis[0,k].plot(listeOptim[k].iloc[:,:nλ])
        axis[0,k].legend(legendλ, loc="upper left")
        axis[0,k].set_title('Evolution λ - ' + typeTaux[k])
        
    
        axis[1,k].plot(listeOptim[k].iloc[:,nλ:-1])
        axis[1,k].legend(legendβ, loc="upper left")
        axis[1,k].set_title('Evolution β - ' + typeTaux[k])  
    
        axis[2,k].plot(listeOptim[k].iloc[:,-1])
        axis[2,k].legend(['Erreur Quad.'], loc="upper right")
        axis[2,k].set_title('Evolution Erreurs - ' + typeTaux[k])  
    
        axis[1,k].sharex(axis[0,k])    
        axis[2,k].sharex(axis[0,k])    

    fig.tight_layout()


    plt.savefig(pp, format='pdf', dpi = 100)
    plt.close('all')


    # Page 2 : Densités
    #--------------------------------------------
    
    fig, axis = plt.subplots(4, 2, dpi = 100, figsize = (13,6))
    
    for k in range(2):
        betas =  listeOptim[k].iloc[:,nλ:-1] 
        betas.columns = legendβ[:nβ]
        dBetas =  betas.diff(axis = 0)
        dBetas.columns = [ 'd' + x for x in legendβ[:nβ]]
        
        fig.subplots_adjust(hspace=0.5)
        axis[0,k].plot(betas)
        axis[0,k].set_title('Evolution Betas - ' + typeTaux[k])
        axis[0,k].grid(True)
    
        axis[1,k].plot(dBetas)
        axis[1,k].set_title('Evolution dBetas - ' + typeTaux[k])   
        axis[1,k].grid(True)
        
        sns.kdeplot(data = betas, ax = axis[2,k])
        axis[2,k].set_title('Densité Betas - ' + typeTaux[k])
    
        sns.kdeplot(data = dBetas, ax = axis[3,k])
        axis[3,k].set_title('Densité dBetas - ' + typeTaux[k])
        
    fig.tight_layout()
    plt.savefig(pp, format='pdf', dpi = 100)
    
    plt.close('all')
    
    
    # Page 3 : Matrice de corrélation et lois
    #--------------------------------------------
    
    #listeBetasG = [x + ' - Tn' for x in legendβ[:nβ]] +  [x + ' - Ti' for x in legendβ[:nβ]] 
       
    #fig, axis = plt.subplots(2, 1, dpi = 150, figsize = (13,6))
   
    #distS = pd.DataFrame()
    #for l in range(len(join_dParams)):
     #   lDist =[list(join_dParams[l].items())[0][1]]
      #  lp =  [round(x, 3) for x in list(join_dParams[l].items())[1][1]]
       # tKS = [round(x, 3) for x in list(join_dParams[l].items())[2][1]]
        #distS= distS.concat([lDist + [lp] + [tKS]], ignore_index=True)
    
   # distS.columns = ['Distribution', 'paramètres', 'p-value K-S']
    #distS.index = listeBetasG
    #distS.reset_index(inplace =True)
    
    #formatTable(axis[0],  distS)
    # axis[0].set(title='Distribution retenues β')
    
    

    
    # Matrice de corrélation observée
   # mask = [[False]*n +[True] *(2*nβ-n) for n in range(1,2*nβ+1)]
    #axis[1].set(title='Matrice de corrélation β')
    #matriceCorrelDf = pd.DataFrame(matriceCorrel, index = listeBetasG, columns = listeBetasG )
    #sns.heatmap(matriceCorrelDf, 
     #           annot = True, 
      #          fmt='.2g',
       #         cmap= 'coolwarm',
                #mask = mask,
        #        ax = axis[1])  
    
    #fig.tight_layout()
    #plt.savefig(pp, format='pdf', dpi = 100)
    
    #plt.close('all')
    
    # Page 4 : Courbes 
    #--------------------------------------------
    
    fig, axis = plt.subplots(2, len(typeTaux), dpi = 150, figsize = (13,6))

    for k in range(len(typeTaux)):
        
        axis[0,k].plot(listeBases[k].T, rasterized=True)
        axis[0,k].set_title('Courbes observées - ' + typeTaux[k])
        
        axis[1,k].plot(listeCourbes[k].T, rasterized=True)
        axis[1,k].set_title('Courbes simulées - ' + typeTaux[k])

    fig.tight_layout()
    plt.savefig(pp, format='pdf')
    plt.close('all')    
    
        
    # Page 4 : Chocs simulés 
    #--------------------------------------------       
    if indicTR :
        fig, axis = plt.subplots(1, 2, dpi = 150, figsize = (13,6))    
          
        axis[0].plot(listeChocs[0].T, rasterized=True)
        axis[0].set_title('Chocs simulées - ' + typeTaux[0]) 
        
        axis[0].plot(listeChocs[2].T, rasterized=True)
        axis[0].set_title('Chocs simulées - ' + typeTaux[1]) 
            
    else :
        fig, axis = plt.subplots(1, 3, dpi = 150, figsize = (13,6))    
    
        for k in range(2):       
            axis[k].plot(listeChocs[k].T, rasterized=True)
            axis[k].set_title('Chocs simulés - ' + typeTaux[k]) 
            
        axis[2].plot(listeChocs[2].T, rasterized=True)
        axis[2].set_title('Chocs simulés - ' + 'Tr')       
            
    fig.tight_layout()
    plt.savefig(pp, format='pdf')
    plt.close('all')    
    
    # Page 4 : Statistiques descritives 
    #--------------------------------------------       
 
    
    for k in range(len(typeTaux)):              
         df = pd.DataFrame(listeBases[k], columns = listeMaturites)
         descSource = df.describe(percentiles = [.01, .05, .10, .90, .95, .99]).round(decimals = 2)
        
         df = pd.DataFrame(listeCourbes[k], columns = listeMaturites)
         descSimule= df.describe(percentiles = [.01, .05, .10, .90, .95, .99]).round(decimals = 2)
         
         fig, axis = plt.subplots(2, 1, dpi = 100, figsize = (13,6))
         
         axis[0].plot(descSource.iloc[1:,:].T)         
         axis[1].plot(descSimule.iloc[1:,:].T)  
         
         for line, name in zip(axis[0].lines, descSource.iloc[1:,:].T.columns):
             y = line.get_ydata()[-1]
             axis[0].annotate(name, xy=(1,y), xytext=(6,0), color=line.get_color(), 
                xycoords = axis[0].get_yaxis_transform(), textcoords="offset points",
                size=14, va="center")
         
         for line, name in zip(axis[1].lines, descSimule.iloc[1:,:].T.columns):
               y = line.get_ydata()[-1]
               axis[1].annotate(name, xy=(1,y), xytext=(6,0), color=line.get_color(), 
                  xycoords = axis[1].get_yaxis_transform(), textcoords="offset points",
                  size=14, va="center")          
            
            
         axis[0].set_title('Taux ' + typeTaux[k] + ' - données sources', y=-0.4)
         axis[1].set_title('Taux ' + typeTaux[k] + ' - données simulées', y=-0.4)
                
         fig.tight_layout()
         plt.savefig(pp, format='pdf')
         plt.close('all')
 
    
         fig, axis = plt.subplots(2, 1, dpi = 100, figsize = (13,6))
    
         descSource = descSource.filter(items=listeMaturitesReduite).reset_index().rename(columns={"index": "Maturités"})
         descSimule = descSimule.filter(items=listeMaturitesReduite).reset_index().rename(columns={"index": "Maturités"})        
    
         formatTable(axis[0], descSource)
         formatTable(axis[1], descSimule)
    
         axis[0].set_title('Taux ' + typeTaux[k] + ' - données sources', y=-0.4)
         axis[1].set_title('Taux ' + typeTaux[k] + ' - données simulées', y=-0.4)
         fig.tight_layout()    
            
         plt.savefig(pp, format='pdf')
         plt.close('all')    
     

   
     
   
    
    
    # Page 5 : Vol de taux initiale vs simules
    #--------------------------------------------
    
    fig, axis = plt.subplots(2, len(typeTaux), dpi = 150, figsize = (13,6))
    
    for k in range(len(typeTaux)):

        axis[0,k].bar(listeMaturites, listeBases[k].std(axis = 0))
        axis[0,k].set_title('Evolution Vol Initiale - ' + typeTaux[k])
            
        axis[1,k].bar(listeMaturites, listeCourbes[k].std(axis = 0))
        axis[1,k].set_title('Evolution Vol Simulée - ' + typeTaux[k]) 
    
    fig.tight_layout()
    plt.savefig(pp, format='pdf')
    plt.close('all')
    
    # Page 6 : Scatters Plot
    
     
    fig, axis = plt.subplots(3, 1, dpi = 100, figsize = (6,10)) 
    
    j=0
    for i in [1,10,30]:
       # plt.colorbar()
        im = axis[j].scatter(x=listeChocs[0][:,i-1], y=listeChocs[1][:,i-1],
                           rasterized=True)
        axis[j].set_xlabel('Taux Nominaux')
        axis[j].set_ylabel('Taux Inflations')
        axis[j].set_title('Dispersion des chocs - Maturité = ' + str(i))
        j = j+1
      
    fig.tight_layout()
    
    
    plt.savefig(pp, format='pdf')
    plt.close('all')
 
    #--------------------------------------------
    
    fig, axis = plt.subplots(3, 1, dpi = 100, figsize = (6,10)) 
    
    j=0
    for i in [1,10,30]:
       # plt.colorbar()
        im = axis[j].scatter(x=listeChocs[0][:,i-1], y=listeChocs[2][:,i-1],
                           rasterized=True)
        axis[j].set_xlabel('Taux Nominaux')
        axis[j].set_ylabel('Taux Réels')
        axis[j].set_title('Dispersion des chocs - Maturité = ' + str(i))
        j = j+1
      
    fig.tight_layout()
    
    
    plt.savefig(pp, format='pdf')
    plt.close('all')
    pp.close()
 

    
    

    
   
    