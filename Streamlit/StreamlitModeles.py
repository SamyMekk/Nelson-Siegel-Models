# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:06:55 2023

@author: smekkaoui
"""


import pandas as pd
import numpy as np
import streamlit as st
from ModelesNS import *



class NS(ModelesNelsonSiegel):
    def __init__(self,b0,b1,b2,lambda1):
        ModelesNelsonSiegel.__init__(self,b0,b1,b2,lambda1)
        
    def Fonctionnelle(self,t):
        first=(1-np.exp(-t/self.lambda1))/(t/self.lambda1)
        second=first-np.exp(-t/self.lambda1)
        return self.b0+self.b1*first+self.b2*second

class NSS(ModelesNelsonSiegel):
    def __init__(self,b0,b1,b2,b3,lambda1,lambda2):
        ModelesNelsonSiegel.__init__(self,b0,b1,b2,lambda1)
        self.b3=b3
        self.lambda2=lambda2
        
        
    def Fonctionnelle(self,t):
        first=(1-np.exp(-t/self.lambda1))/(t/self.lambda1)
        second=first-np.exp(-t/self.lambda1)
        third=(1-np.exp(-t/self.lambda2))/(t/self.lambda2)-np.exp(-t/self.lambda2)
        return self.b0+self.b1*first+self.b2*second+self.b3*third
        
    
class NSSF(ModelesNelsonSiegel):
    def __init__(self,b0,b1,b2,b3,lambda1,lambda2):
        ModelesNelsonSiegel.__init__(self,b0,b1,b2,lambda1)
        self.b3=b3
        self.lambda2=lambda2
        
    def Fonctionnelle(self,t):
        first=(1-np.exp(-t/self.lambda1))/(t/self.lambda1)
        second=first-np.exp(-t/self.lambda1)
        third=pow(t/(self.lambda2),2)*np.exp(-t/self.lambda2)
        return self.b0+self.b1*first+self.b2*second+self.b3*third
        

    
        
        
        
        
        
    
# Partie Streamlit dans le code 


st.title(''' Application Simple faite par  Samy Mekkaoui  pour la Modélisation de courbes de taux par les modèles NS''')



option=st.selectbox("Choississez le type de modèle que vous voulez étudier",
                        ('NS','NSS','NSSF'))


def user_input(option):
    if option=="NS":
        b0=st.sidebar.number_input("Choississez la valeur de  β0 ",value= 0.03)
        b1=st.sidebar.number_input("Choississez la valeur de  β1 ",value= -0.01)
        b2=st.sidebar.number_input("Choississez la valeur de  β2 ",value= 0.02)
        lambda1=st.sidebar.number_input("Choississez la valeur de  λ1 ",value= 0.5)
        data={'''β0''': b0,
          '''β1''':b1,
          '''β2''':b2,
          '''λ1''':lambda1}
        Parametres=pd.DataFrame(data,index=["Paramètres"])
        return Parametres
    if option=="NSS":
        b0=st.sidebar.number_input("Choississez la valeur de  β0 ",value= 0.03)
        b1=st.sidebar.number_input("Choississez la valeur de  β1 ",value= -0.01)
        b2=st.sidebar.number_input("Choississez la valeur de  β2 ",value= 0.02)
        b3=st.sidebar.number_input("Choississez la valeur de  β3 ",value= 0.01)
        lambda1=st.sidebar.number_input("Choississez la valeur de  λ1 ",value= 0.5)
        lambda2=st.sidebar.number_input("Choississez la valeur de  λ2 ",value= 5.5)
        data={'''β0''': b0,
          '''β1''':b1,
          '''β2''':b2,
          '''β3''':b3,
          '''λ1''':lambda1,
          '''λ2''':lambda2}
        Parametres=pd.DataFrame(data,index=["Paramètres"])
        return Parametres
    if option=="NSSF":
        b0=st.sidebar.number_input("Choississez la valeur de  β0 ",value= 0.03)
        b1=st.sidebar.number_input("Choississez la valeur de  β1 ",value= -0.01)
        b2=st.sidebar.number_input("Choississez la valeur de  β2 ",value= 0.02)
        b3=st.sidebar.number_input("Choississez la valeur de  β3 ",value= 0.01)
        lambda1=st.sidebar.number_input("Choississez la valeur de  λ1 ",value= 0.5)
        lambda2=st.sidebar.number_input("Choississez la valeur de  λ2 ",value= 5.5)
        data={'''β0''': b0,
          '''β1''':b1,
          '''β2''':b2,
          '''β3''':b3,
          '''λ1''':lambda1,
          '''λ2''':lambda2}
        Parametres=pd.DataFrame(data,index=["Paramètres"])
        return Parametres
   
    
   
df=user_input(option)

st.header("Vous avez sélectionné le modèle " + str(option) + " dont on rappelle la dynamique ci-dessous")



if option=="NS":
    st.latex("r(t)=β_{0}+β_{1}\\frac{(1-e^{-\\frac{t}{\lambda}})}{\\frac{t}{\lambda}}++β_{2}(\\frac{(1-e^{-\\frac{t}{\lambda}})}{\\frac{t}{\lambda}}-e^{-\\frac{t}{\lambda}})")
    st.subheader("Avec :")
    st.write("$β_{0}$ : le taux de long terme")
    st.write("$β_{1}$ : le facteur de pentification")
    st.write("$β_{2}$ : le facteur de courbure")
    st.write("$\lambda$ : le facteur temporel associé à la courbure")


if option=="NSS":
    st.latex("r(t)=β_{0}+β_{1}\\frac{(1-e^{-\\frac{t}{\lambda_{1}}})}{\\frac{t}{\lambda_{1}}}+β_{2}(\\frac{(1-e^{-\\frac{t}{\lambda_{1}}})}{\\frac{t}{\lambda_{1}}}-e^{-\\frac{t}{\lambda_{1}}})+β_{3}(\\frac{(1-e^{-\\frac{t}{\lambda_{2}}})}{\\frac{t}{\lambda_{2}}}-e^{-\\frac{t}{\lambda_{2}}})")
    st.subheader("Avec :")
    st.write("$β_{0}$ : le taux de long terme")
    st.write("$β_{1}$ : le facteur de pentification")
    st.write("$β_{2}$ : le premier facteur de courbure")
    st.write("$β_{3}$ : le second facteur de courbure")
    st.write("$\lambda_{1}$ : le facteur temporel associé à la première courbure")
    st.write("$\lambda_{2} $ : le facteur temporel associé à la seconde courbure")    



elif option=="NSSF":
    st.latex("r(t)=β_{0}+β_{1}\\frac{(1-e^{-\\frac{t}{\lambda_{1}}})}{\\frac{t}{\lambda_{1}}}+β_{2}(\\frac{(1-e^{-\\frac{t}{\lambda_{1}}})}{\\frac{t}{\lambda_{1}}}-e^{-\\frac{t}{\lambda_{1}}})+β_{3}((\\frac{t}{\lambda_{2}})^{2}e^{-\\frac{t}{\lambda_{2}}})")
    st.subheader("Avec :")
    st.write("$β_{0}$ : le taux de long terme")
    st.write("$β_{1}$ : le facteur de pentification")
    st.write("$β_{2}$ : le premier facteur de courbure")
    st.write("$β_{3}$ : le second facteur de courbure")
    st.write("$\lambda_{1}$ : le facteur temporel associé à la première courbure")
    st.write("$\lambda_{2} $ : le facteur temporel associé à la seconde courbure")    




st.subheader('Voici les paramètres du modèle que vous avez choisi')

st.write(df)


def user_input2():
    Horizon=st.number_input("Choississez l'horizon de projection que vous souhaitez de la courbe'",value=30)
    data={'Horizon de Projection': Horizon}
    Parametres2=pd.DataFrame(data,index=[0])
    return Parametres2


df2=user_input2()
st.write(df2)

T=df2['Horizon de Projection'][0]

if option=="NS":
    b0=df['β0'][0]
    b1=df['β1'][0]
    b2=df['β2'][0]
    lambda1=df['λ1'][0]
    Modele=NS(b0,b1,b2,lambda1)
else:
    b0=df['β0'][0]
    b1=df['β1'][0]
    b2=df['β2'][0]
    b3=df['β3'][0]
    lambda1=df['λ1'][0]
    lambda2=df['λ2'][0]
    if option=="NSS":
        Modele=NSS(b0,b1,b2,b3,lambda1,lambda2)
    elif option=="NSSF":
        Modele=NSSF(b0,b1,b2,b3,lambda1,lambda2)
        
st.pyplot(Modele.DiffusionTaux(T))

    