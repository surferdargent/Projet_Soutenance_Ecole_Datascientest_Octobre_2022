import streamlit as st
import pandas as pd 
import numpy as np 



# import y_test,X_test,y_train,split_normalisation,mean_rolling,creat_df

from PIL import Image
from datetime import datetime 





title = "Stratégie et Conclusion"
sidebar_name = "Stratégie et Conclusion"


def run():

    st.title(title)

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
    #data[['Bkm_prediction','Victoire_reel','Winner']]
    data["Bkm_predict_vict"] = data['Bkm_prediction'].replace({"D":0,"V":1}).astype(float)

    
    col1,col2= st.columns([1,1])

    with col1:
        #st.markdown(
        mise_de_depart = st.number_input('Mise de départ',1,100,10, key = 2)
        
        start_date, end_date = st.date_input('Choisir date de début, date de fin :', [datetime(2016,1,1),datetime(2018,3,4)],key=6)
        
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

    #st.dataframe(data)
    
    
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
    

    # Option 2 : stratégie avec pari uniquement sur la prédiction du modèle pour les gagnants 

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
    date_split = pd.Timestamp(2016, 1, 1)
    data_var=  data_var[data_var['Year'] >= date_split]
    data_var['Year'] = pd.to_datetime(data_var['Year'])
    data_var['Year']= data_var['Year'].dt.date
    data.reset_index(drop=True, inplace=True)
    data_var = data_var.sort_values(by=["Year"],ascending = True)
    start_date_var, end_date_var = start_date, end_date
    
    if start_date_var < end_date_var:
        pass
    
    else:
        st.error('Error: Date de fin doit être choisi après la date de début.')
    
  
    mask_var = (data_var['Year'] >= start_date_var) & (data_var['Year'] <= end_date_var)
    data_var_mask = data_var.loc[mask_var]
    

    
    def paris1(mise_de_dep=10):
        mise_de_depart = mise_de_dep
        data_var_mask["Gains"] =data_var_mask["Win"] * mise_de_depart * (data_var_mask['B365']-1)
        data_var_mask["Mise"] = data_var_mask["Win"] * mise_de_depart
       
        st.dataframe(data_var_mask)
        # df_lines = df.apply(np.sum ,axis = 0) 
        st.write ("La somme pariée serait de 116070 euros et le gain prédit de 49807 euros si nous siuvons les recommandations des bookmakers sur notre jeu de test .Soit 43.0 % de bénéfices")
                
        # return st.write("La somme pariée serait de", mise2, "euros et le gain prédit de", gain2,"euros."),
        # st.write("Soit",round( gain2/mise2,2)*100,"% de bénéfices")
        
    paris1()
    
    # Option 3 : stratégie 
    
    st.markdown(
        """
          ## Troisième stratégie          
        """
    )
    
    st.markdown(
        """
        Stratégie de sélection des mises appuyées sur les prédictions de notre modèle
        intégrant un seuil de sécurité.
        Cette troisième stratégie repose également sur notre modèle mais limite l’engagement
        d’une mise aux situations dans lesquels le modèle à une confiance de plus de 80% dans sa
        prédiction. Cette stratégie, que l’on peut qualifier de prudente, vise à minimiser les risques
        et diminuer le montant des sommes engagées.
         La somme pariée serait de 20600 euros et le gain prédit de 7843 euros.Soit 38.0 % de bénéfices. 
         
         
         La dernière stratégie serait la plus optimale car le bénéfice est de 38,0 % légèrement supérieur au bénéfice des préconisations bookmakers ( 36,0 %) 
         Surtout la somme engagée pour la dernière stratégie est de 20600 euros contre 40970 euros si on suit les préconisations bookmakersNotre stratégie battrait les bookmakers ...""")
      
    
    """
    Pour le test nous allons prendre les 10 derniers joueurs de notre jeu de test et appliquer notre  stratégie 
            
    """   
    
  
    @st.cache()   
    def demo(taux,mse_depart ):
        #list_proba = []
        
        # gain = 0
        # mise_totale = 0,
        # mise_de_depart = mse_depart
        # seuil = taux
        # # y_pred_proba = probs
        # for i,probas in enumerate (y_pred_proba):
        #     cotes = data_var['B365'].iloc[i]
        #     if probas[1] >= seuil :
                
        #         if y_test.iloc[i]== 1:
        #             gain += round((mise_de_depart * ( probas[1] - seuil ) / ( 1 - seuil )) * (cotes - 1))
        #         mise_totale += round(mise_de_depart * ( probas[1] - seuil ) / ( 1 - seuil ))
    #     #st.write("La somme pariée serait de", mise_totale, "euros et le gain prédit de", gain,"euros.")
    #     #st.write("Soit",round( gain/mise_totale,2)*100,"% de bénéfices")
    #     ###return proba[1],
    
    
     _left, _right = st.columns(2)
    
     with _left:
        taux  = st.number_input('Sélectionner un taux entre 51 et 100 %',51,100,51,key=5)
        mse_depart = st.number_input('Sélectionner votre mise de départ',1,20000,10,key=3)
    
     with _left:
        if st.button('Lancer la démo'):
            demo(taux,mse_depart)
            st.markdown("""
                      """)       

    
    st.markdown("---")
    
    st. markdown(
        
            """
            ## Conclusion
            
            """
        )
    st.markdown(
    """
        Au terme de notre projet, nous avons pu entraîner un modèle de prévision des résultats des
        matchs, plus performant que les prédictions des bookmakers. La qualité de cette information nous a permis de développer une stratégie de paris maximisant nos gains,
        par rapport à une situation dans laquelle nous n’aurions pas été en mesure de prédire avec une
        confiance suffisante.
        Avec davantage de temps et forts de l’expérience acquise lors du projet, il nous apparaît
        aujourd’hui que nous aurions pu explorer davantage la piste suivante.Lors de la phase exploratoire des données 
        nous avons produit deux types de statistiques qui auraient pu éventuellement servir à élaborer une stratégie de paris plus efficace.
        Nous aurions ainsi pu établir d’une part des statistiques de performance des joueurs selon
         d’autres paramètres (surface, tournoi, adversaire...), et d’autre part identifier les types de
        matchs pour lesquels les pronostics des bookmakers sont moins performants. Cela aurait
        pu nous permettre de mener des stratégies de niche exploitant les défaillances des
        bookmakers.
    """
    )
   

# Stratégie

#  #title = "Stratégie"
#  sidebar_name = "Stratégie et Conclusion"

#  st.title("Stratégie et Conclusion")

 