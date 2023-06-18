import pandas as pd 
import seaborn as sns 
import numpy  as np 
import streamlit as st 

sns.set_theme()  

from PIL import Image

from datetime import datetime 
import seaborn as sns 


#title = "Stratégie"
sidebar_name = "Stratégie et Conclusion"


def run():
    
    
    st.title("Stratégie et Conclusion")
   
    st.markdown(
    """
    Nous avons testé trois stratégies différentes et évaluer laquelle permet de maximiser le
    profit. Pour ces trois stratégies notre mise de départ est 10 euros par défaut, et nous établissons les gains
    obtenus sur l’ensemble des matchs qui se sont déroulés entre 01 janvier 2016 au 04 mars
    2018, soit notre échantillon de test.
      """
          )
    # Option 1  : stratégie naive 
    data = pd.read_csv('atp_data.csv',parse_dates=['Date'])
    data["Date"] = pd.to_datetime(data["Date"])
    data['Date'] = data['Date'].dt.date
    data.rename(columns={'Date': 'Year'}, inplace=True)
    
    
    
    # Synthèse des prévisions des bookmakers dans un dataframe 

    data['Bkm_prediction'] = data[['B365W','B365L']].apply(lambda x: np.argmin(x), axis=1)
    data['Bkm_prediction'] = data['Bkm_prediction'].replace(to_replace=[0, 1], value=['V', 'D'])


    # Transformer les valeurs de la variable Winner en "V" comme victoire pour comparer les prév et le réel
    data['Victoire_reel'] = "V"
    data[['Bkm_prediction','Victoire_reel','Winner']]
    data["Bkm_predict_vict"] = data['Bkm_prediction'].replace({"D":0,"V":1}).astype(float)

    
    col1,col2= st.columns([1,1])

    with col1:
        #st.markdown(
        mise_de_depart = st.number_input('Mise de départ',1,100,10)
        
        start_date, end_date = st.date_input('Choisir date de début, date de fin :', [datetime(2016,1,1),datetime(2018,3,3)])
        
        if start_date < end_date:
            pass
        # else:
        #     st.error('Error: Date de fin doit être choisi après la date de début.')
        
        #greater than the start date and smaller than the end date  
            
            
            mask = (data['Year'] > start_date) & (data['Year'] <= end_date)
            data = data.loc[mask]
           
       
    
    with col2:
       
        st.image(Image.open("assets/Argent2.png"))
       
    # st.dataframe(data)
    # # mask = (data['Year'] > start_date) & (data['Year'] <= end_date)
    # # data = data.loc[mask]
   
    st.dataframe(data)
    
    
    st.markdown(
        """
          ## Première stratégie          
        """
    )
    

    st.markdown(
        """
          Une stratégie dite naive car nous alignons nos paris sur les cotes des bookmakers. 
         
        """
    )
    

     
    data["Mise"] =  data['Bkm_predict_vict'] * mise_de_depart
    data["Gain"] =  data["Mise"] * (data["B365W"] -1 )
    
    st.write("La somme pariée serait de", round(data["Mise"].sum(axis=0),2), "euros et le gain prédit de", round(data["Gain"].sum(axis=0),2),"euros.")
    st.write("Soit",round( round(data["Gain"].sum(axis=0),2)/round(data["Mise"].sum(axis=0)),2)*100,"% de bénéfices.Ce gain représente la somme gagnée si nous suivons les préco du bookmaker B365.")
    
   
   
   
    st.markdown(
        """
          ## Deuxième stratégie          
        """
    )
    
    st.markdown(
        """
         Une stratégie où nous misons uniquement sur les victoires prédites par notre modèle 
         
        """
    )
    
    # Base pour récupérer les cotes
    data_var = pd.read_csv('df_variables_enrichies.csv',parse_dates=['Year'])
    data_var['Year'] = pd.to_datetime(data_var['Year'])
    data_var['Year']= data_var['Year'].dt.date
    start_date_var, end_date_var = start_date, end_date
    
    if start_date_var < end_date_var:
        pass
    # else:
    #     st.error('Error: Date de fin doit être choisi après la date de début.')
    
    #greater than the start date and smaller than the end date
    mask_var = (data_var['Year'] > start_date_var) & (data_var['Year'] <= end_date_var)
    data_var = data_var.loc[mask_var]
    
   
    #st.dataframe(data_var)
   
    

    # Option 2 : stratégie avec pari uniquement sur la prédiction du modèle pour les gagnants 
    

    """
  
    
    def paris1(gain = 0, mise_totale = 0 , mise_de_depart = mise_de_depart ):
        for i in range ( 1 , len ( new_y_test)):
            cotes = data_var['B365'].iloc[i] 
            if new_y_test.iloc[i]== 1: 
                gain += round(mise_de_depart  * (cotes - 1))
            mise_totale += round(mise_de_depart)
        st.write("La somme pariée serait de", mise_totale, "euros et le gain prédit de", gain,"euros.")
        st.write("Soit",round( gain/mise_totale,2)*100,"% de bénéfices")
"""

    
    # # Base pour récupérer les cotes
    # data = pd.read_csv('df_variables_enrichies.csv',parse_dates=['Year'])
  
    # data["Year"] = pd.to_datetime(data["Year"])
    # data['Year'] = data['Year'].dt.date
    # date_split = pd.Timestamp(2016, 1, 1)
    
    # new_df_strategie_test = data.sort_values(by=["Year"],ascending = True)
    # new_df_strategie_test =  data[data['Year'] >= date_split]
    
    
    
    
    
    # 
    
    # import_y_test = third_tab.y_test
    # y_test = third_tab.optimisation_models(import_y_test)
    # def paris1(gain = 0, mise_totale = 0 , mise_de_depart = 10 ):
       
    #     for i in range ( 1 , len (y_test)):
    #        cotes = new_df_strategie_test['B365'].iloc[i] 
    #        if y_test.iloc[i]== 1: 
    #           gain += round(mise_de_depart  * (cotes - 1))
    #        mise_totale += round(mise_de_depart)
    #        st.write("La somme pariée serait de", mise_totale, "euros et le gain prédit de", gain,"euros.")
    #        st.write("Soit",round( gain/mise_totale,2)*100,"% de bénéfices")


    # paris1()
    
    st.markdown("---")
    
st. markdown(
    
        """
        ## Conclusion
        
        """
    )
st.markdown(
"""
    Au terme de notre projet, nous avons pu entraîner un modèle de prévision des résultats des
    matchs plus performant que les prédictions des bookmakers. La qualité de cette
    information nous a permis de développer une stratégie de paris maximisant nos gains, par
    rapport à une situation dans laquelle nous n’aurions pas été en mesure de prédire avec une
    confiance suffisante.
    Avec davantage de temps et forts de l’expérience acquise lors du projet, il nous apparaît
    aujourd’hui que nous aurions pu explorer davantage la piste suivante :
    Lors de la phase exploratoire des données, nous avons produit deux types de statistiques
    qui auraient pu éventuellement servir à élaborer une stratégie de paris plus efficace. Nous
    aurions ainsi pu établir d’une part des statistiques de performance des joueurs selon
    d’autres paramètres (surface, tournoi, adversaire...), et d’autre part identifier les types de
    matchs pour lesquels les pronostics des bookmakers sont moins performants. Cela aurait
    pu nous permettre de mener des stratégies de niche exploitant les défaillances des
    bookmakers.
"""
)
    