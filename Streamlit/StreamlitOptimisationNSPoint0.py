
"""
Created on Tue May 16 16:44:21 2023

@author: smekkaoui
"""


import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import streamlit 


streamlit.header("Application simple comme optimiseur de paramètres des modèles NS avec Contrainte et Choix du Taux court Terme")




option=streamlit.selectbox("Choississez le type de modèle que vous voulez étudier",
                        ('NS','NSS','NSSF'))


streamlit.subheader("Vous avez choisi une optimisation avec le modèle " + str(option))

def user_input():
        Taux0=streamlit.sidebar.number_input("Choississez la valeur du Taux très court-terme ",value= 0.013)
        Taux2=streamlit.sidebar.number_input("Choississez la valeur du Taux 2 ans ",value= 0.018)
        Taux5=streamlit.sidebar.number_input("Choississez la valeur du Taux 5 ans ",value= 0.027)
        Taux10=streamlit.sidebar.number_input("Choississez la valeur du Taux 10 ans ",value= 0.033)
        Taux20=streamlit.sidebar.number_input("Choississez la valeur du Taux 20 ans",value= 0.034)
        data={1/52:Taux0,
            2: Taux2,
          5:Taux5,
          10:Taux10,
          20:Taux20}
        Parametres=pd.DataFrame(data,index=["Taux"]).T
        return Parametres
    
def user_input2():
    Contrainteb0=streamlit.sidebar.number_input("Choississez le range de variation de β0 ( par exemple +/- 1% autour du Taux 20 ans) ",value= 0.01)
    Contrainteb1=streamlit.sidebar.number_input("Choississez le range de variation de β1 ( par exemple +/- 1% autour de la diff entre Taux très court terme et Taux 20 ans) ",value= 0.005)
    if option=="NS":
        Contraintelambda=streamlit.sidebar.number_input("Choississez la borne maximale sur λ",value=30)
        data={"Contrainte β0":Contrainteb0,
              "Contrainte β1":Contrainteb1,
              "Contrainte max λ":Contraintelambda
             }
        Parametres=pd.DataFrame(data,index=["Contraintes"]).T
        return Parametres
    else:
        Contraintelambda1=streamlit.sidebar.number_input("Choississez la borne maximale sur λ1",value=5)
        Contraintelambda2min=streamlit.sidebar.number_input("Choississez la borne minimale sur λ2",value=5)
        Contraintelambda2max=streamlit.sidebar.number_input("Choissisez la borne maximale sur λ2",value=30)
        data={"Contrainte β0":Contrainteb0,
          "Contrainte β1":Contrainteb1,
          "Contrainte max λ1":Contraintelambda1,
          "Contrainte min λ2":Contraintelambda2min,
          "Contrainte max λ2":Contraintelambda2max}
        Parametres=pd.DataFrame(data,index=["Contraintes"]).T
        return Parametres
    
    
df=user_input()
df2=user_input2()

streamlit.subheader('Voici les Taux que vous avez choisis en Input : ')

streamlit.write(df)
streamlit.write(df2)


# Définition des fonctionnelles 

def modeleNS(b0: float,b1: float,b2: float,lambda1: float,T: float):
    first=(1-np.exp(-T/lambda1))/(T/lambda1)
    second=first-np.exp(-T/lambda1)
    return b0+b1*first+b2*second
    
def modeleNSS(b0 : float, b1: float , b2: float , b3 : float, lambda1 : float, lambda2: float , T: float):
    first=(1-np.exp(-T/lambda1))/(T/lambda1)
    second=first-np.exp(-T/lambda1)
    third=(1-np.exp(-T/lambda2))/(T/lambda2)-np.exp(-T/lambda2)
    return b0+b1*first+b2*second+b3*third
    
def modeleNSSF(b0: float, b1:float, b2:float , b3 : float, lambda1: float, lambda2: float, T: float):
    first=(1-np.exp(-T/lambda1))/(T/lambda1)
    second=first-np.exp(-T/lambda1)
    third=pow(T/lambda2,2)*np.exp(-T/lambda2)
    return b0+b1*first+b2*second+b3*third

# Partie Optimisation 
def obj1(arguments,data,modele):
    if modele=="NS":
        b0,b1,b2,lambda1=arguments
        somme=0
        Time=list(data.index)
        for element in Time:
            somme+=pow(modeleNS(b0,b1,b2,lambda1,element)-data["Taux"][element],2)     
        return somme
    elif modele=="NSS":
        b0,b1,b2,b3,lambda1,lambda2=arguments
        somme=0
        Time=list(data.index)
        for element in Time:
            somme+=pow(modeleNSS(b0,b1,b2,b3,lambda1,lambda2,element)-data["Taux"][element],2)     
        return somme
    elif modele=="NSSF":
        b0,b1,b2,b3,lambda1,lambda2=arguments
        somme=0
        Time=list(data.index)
        for element in Time:
            somme+=pow(modeleNSSF(b0,b1,b2,b3,lambda1,lambda2,element)-data["Taux"][element],2)     
        return somme
    
    
def minimisation(data,modele):
    Contrainteb0=float(df2.transpose()["Contrainte β0"][0])
    Contrainteb1=float(df2.transpose()["Contrainte β1"][0])
    if modele=="NS":
        Contraintelambda=float(df2.transpose()["Contrainte max λ"][0])
        first_guess=[0.02,0.01,0.01,2]
        bounds=((data["Taux"][data.index[-1]]-Contrainteb0,data["Taux"][data.index[-1]]+Contrainteb0),(data["Taux"][data.index[0]]-data["Taux"][data.index[-1]]-Contrainteb1,data["Taux"][data.index[0]]-data["Taux"][data.index[-1]]+Contrainteb1),(-0.5,0.5),(0.1,Contraintelambda))
        minimisation=minimize(obj1,first_guess,args=(data,modele),method="Nelder-Mead",bounds=bounds,options={"maxiter":100000})
        b0,b1,b2,lambda1=minimisation.x
        dictParam={'β0':b0,
              'β1':b1,
              'β2':b2,
              'λ1':lambda1,}
        Data=pd.DataFrame(dictParam,index=["Résultats Paramètres Opti"])
        return Data
    if modele=="NSS" or modele=="NSSF":
        Contraintelambda1=float(df2.transpose()["Contrainte max λ1"][0])
        Contrainteminlambda2=float(df2.transpose()["Contrainte min λ2"][0])
        Contraintemaxlambda2=float(df2.transpose()["Contrainte min λ2"][0])
        first_guess=[0.02,0.01,0.0001,0.0003,0.12,2]
        bounds=((data["Taux"][data.index[-1]]-Contrainteb0,data["Taux"][data.index[-1]]+Contrainteb0),(data["Taux"][data.index[0]]-data["Taux"][data.index[-1]]-Contrainteb1,data["Taux"][data.index[0]]-data["Taux"][data.index[1]]+Contrainteb1),(-0.5,0.5),(-0.5,0.5),(0.1,Contraintelambda1),(Contrainteminlambda2,Contraintemaxlambda2))
        minimisation=minimize(obj1,first_guess,args=(data,modele),method="Nelder-Mead",bounds=bounds,options={"maxiter":100000})
        b0,b1,b2,b3,lambda1,lambda2=minimisation.x
        dictParam={'β0':b0,
              'β1':b1,
              'β2':b2,
              'β3':b3,
              'λ1':lambda1,
              'λ2':lambda2,}
        Data=pd.DataFrame(dictParam,index=["Résultats Paramètres Opti"])
        return Data
     



def DataTauxNS(b0: float, b1:float,b2:float, lambda1:float):
    Durée=[1/52,2,5,10,20]
    L=[]
    for element in Durée :
        L.append(modeleNS(b0,b1,b2,lambda1,element))
    dictN = {'Temps' : Durée, 'Taux' : L}
    TauxN = pd.DataFrame.from_dict(data = dictN)
    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    ax.plot(TauxN["Temps"],TauxN["Taux"])
    ax.plot(TauxN["Temps"],df["Taux"])
    ax.legend(["Modèle","Marché"])
    ax.set_ylabel("Taux d'intéret")
    ax.set_title("Représentation des courbes")
    ax.set_xlabel("Temps en années")
    return TauxN,fig


def DataTauxNS2(b0: float, b1:float,b2:float, lambda1:float):
    Durée = [x/12 for x in range(0,481)] 
    DuréeM=[x for x in range (0,481)]
    L=[]
    for element in Durée :
        L.append(modeleNS(b0,b1,b2,lambda1,element))
    dictN = {'Temps' : DuréeM, 'Taux' : L}
    TauxN = pd.DataFrame.from_dict(data = dictN)
    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    ax.plot(TauxN["Temps"],TauxN["Taux"])
    return TauxN,fig
  
def DataTauxNSS(b0: float, b1:float,b2:float,b3:float ,lambda1:float,lambda2:float):
    Durée=[1/52,2,5,10,20]
    L=[]
    for element in Durée :
        L.append(modeleNSS(b0,b1,b2,b3,lambda1,lambda2,element))
    dictN = {'Temps' : Durée, 'Taux' : L}
    TauxN = pd.DataFrame.from_dict(data = dictN)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(TauxN["Temps"],TauxN["Taux"])
    ax.plot(TauxN["Temps"],df["Taux"])
    ax.legend(["Modèle","Marché"])
    ax.set_ylabel("Taux d'intéret")
    ax.set_title("Représentation des courbes")
    ax.set_xlabel("Temps en années")

    return TauxN,fig

def DataTauxNSS2(b0: float, b1:float,b2:float,b3:float ,lambda1:float,lambda2:float):
    Durée = [x/12 for x in range(0,481)]
    DuréeM=[x for x in range(0,481)]
    L=[]
    for element in Durée :
        L.append(modeleNSS(b0,b1,b2,b3,lambda1,lambda2,element))
    dictN = {'Temps' : DuréeM, 'Taux' : L}
    TauxN = pd.DataFrame.from_dict(data = dictN)
    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    ax.plot(TauxN["Temps"],TauxN["Taux"])
    return TauxN,fig
  
def DataTauxNSSF(b0: float, b1:float,b2:float,b3:float, lambda1:float,lambda2:float):
    Durée=[1/52,2,5,10,20]
    L=[]
    for element in Durée :
        L.append(modeleNSSF(b0,b1,b2,b3,lambda1,lambda2,element))
    dictN = {'Temps' : Durée, 'Taux' : L}
    TauxN = pd.DataFrame.from_dict(data = dictN)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(TauxN["Temps"],TauxN["Taux"])
    ax.plot(TauxN["Temps"],df["Taux"])
    ax.legend(["Modèle","Marché"])
    ax.set_ylabel("Taux d'intéret")
    ax.set_title("Représentation des courbes")
    ax.set_xlabel("Temps en années")
    return TauxN,fig

def DataTauxNSSF2(b0: float, b1:float,b2:float,b3:float ,lambda1:float,lambda2:float):
    Durée = [x/12 for x in range(0,481)] 
    DuréeM=[x for x in range (0,481)]
    L=[]
    for element in Durée :
        L.append(modeleNSSF(b0,b1,b2,b3,lambda1,lambda2,element))
    dictN = {'Temps' : DuréeM, 'Taux' : L}
    TauxN = pd.DataFrame.from_dict(data = dictN)
    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    ax.plot(TauxN["Temps"],TauxN["Taux"])
    return TauxN,fig


# Passage au Format Excel
def to_excel(df1,df2,df3,name):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df1.to_excel(writer, index=False, sheet_name=name)
    df2.to_excel(writer,index=True,sheet_name=name,startcol=3)
    df3.to_excel(writer,index=True,sheet_name=name,startcol=3,startrow=3)
    workbook = writer.book
    worksheet = writer.sheets[name]
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    processed_data = output.getvalue()
    return processed_data


SheetName=streamlit.text_input("Choississez le nom de la feuille Excel en Sortie")

if streamlit.button("Cliquer sur le bouton pour lancer l'analyse"):
    
    if option=="NS":
        ResultatOpti=minimisation(df,option)
        streamlit.subheader("Voici les Résultats issus de l'optimisation")
        streamlit.write(ResultatOpti)
        b0,b1,b2,lambda1=ResultatOpti['β0'][0],ResultatOpti['β1'][0],ResultatOpti['β2'][0],ResultatOpti['λ1'][0]
        CourbeTaux,fig=DataTauxNS(b0,b1,b2,lambda1)
        CourbeTaux2,fig2=DataTauxNS2(b0,b1,b2,lambda1)
        streamlit.subheader("Voici l'adéquation de la courbe marché et de la courbe modèle obtenue avec les 4 points 2,5,10 et 20 ans : ")
        streamlit.pyplot(fig)
        streamlit.subheader("Voici la courbe obtenue jusqu'à 40 ans avec les paramètres du modèle : ")
        streamlit.pyplot(fig2)
        Excel=to_excel(CourbeTaux2,ResultatOpti,df2,SheetName)
        streamlit.download_button(label="Télécharger la courbe " +  str(option) + " au format xlsx",data=Excel,file_name="CourbeTauxNS.xlsx")
    if option=="NSS":
        ResultatOpti=minimisation(df,option)
        streamlit.subheader("Voici les Résultats issus de l'optimisation")
        streamlit.write(ResultatOpti)
        b0,b1,b2,b3,lambda1,lambda2=ResultatOpti['β0'][0],ResultatOpti['β1'][0],ResultatOpti['β2'][0],ResultatOpti['β3'][0],ResultatOpti['λ1'][0],ResultatOpti['λ2'][0]
        CourbeTaux,fig=DataTauxNSS(b0,b1,b2,b3,lambda1,lambda2)
        CourbeTaux2,fig2=DataTauxNSS2(b0,b1,b2,b3,lambda1,lambda2)
        streamlit.subheader("Voici l'adéquation de la courbe marché et de la courbe modèle obtenue avec les 4 points 2,5,10 et 20 ans : ")
        streamlit.pyplot(fig)
        streamlit.subheader("Voici la courbe obtenue jusqu'à 40 ans avec les paramètres du modèle : ")
        streamlit.pyplot(fig2)
        Excel=to_excel(CourbeTaux2,ResultatOpti,df2,SheetName)
        streamlit.download_button(label="Télécharger la courbe " +  str(option) + " au format xlsx",data=Excel,file_name="CourbeTauxNSS.xlsx")
    if option=="NSSF":
        ResultatOpti=minimisation(df,option)
        streamlit.subheader("Voici les Résultats issus de l'optimisation")
        streamlit.write(ResultatOpti)
        b0,b1,b2,b3,lambda1,lambda2=ResultatOpti['β0'][0],ResultatOpti['β1'][0],ResultatOpti['β2'][0],ResultatOpti['β3'][0],ResultatOpti['λ1'][0],ResultatOpti['λ2'][0]
        CourbeTaux,fig=DataTauxNSSF(b0,b1,b2,b3,lambda1,lambda2)
        CourbeTaux2,fig2=DataTauxNSSF2(b0,b1,b2,b3,lambda1,lambda2)
        streamlit.subheader("Voici l'adéquation de la courbe marché et de la courbe modèle obtenue avec les 4 points 2,5,10 et 20 ans : ")
        streamlit.pyplot(fig)
        streamlit.subheader("Voici la courbe obtenue jusqu'à 40 ans avec les paramètres du modèle : ")
        streamlit.pyplot(fig2)
        Excel=to_excel(CourbeTaux2,ResultatOpti,df2,SheetName)
        streamlit.download_button(label="Télécharger la courbe " +  str(option) + " au format xlsx",data=Excel,file_name="CourbeTauxNSSF.xlsx")
    
        
      


        



