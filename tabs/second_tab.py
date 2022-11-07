import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#from traitement import y_test,X_test,y_train,split_normalisation,mean_rolling,creat_df






title = "Datavisualisation"
sidebar_name = "Datavisualisation"


def run():

    st.title(title)

    st.markdown(
        """
        ## Quelques infos sur le jeu de données...
        """
    )
    
   
    st.markdown(
        """
        Cette partie est consacrée à la présentation des données, afin de comprendre la
        structure et la nature des informations sur lesquelles s'appuient notre étude.
        Le jeu de données contient des informations sur les résultats obtenus dans les tournois
        de tennis [ATP](https://www.atptour.com/) entre janvier 2000 et mars 2018. Comme le montre le dataframe ci-dessous,
        chaque ligne indique pour un tournoi donné le nom du vainqueur, les principales
        caractéristiques de ce tournoi et les cotes estimées par les bookmakers (colonnes "PSW" jusqu’à "B365L").
        Nous avons aussi dans la colonne "Comment" des indications sur la façon dont chaque match s'est terminé( disqualification, abandon,...) 
        
        """
    )
   
   
    @st.cache(suppress_st_warning=True,allow_output_mutation=True)
    def load_data():
        data = pd.read_csv('atp_data.csv',parse_dates=['Date'])
        data["Date"] = pd.to_datetime(data["Date"])
        data['Date'] = data['Date'].dt.date
        return data
    data = load_data()

    st.dataframe(data.head())
   
    st.markdown(
        """
        :tennis: La valeur elo :
   
        """
    )
    
    st.markdown(
        """
        Il s’agit d’une donnée ajoutée par le créateur du dataset permettant de mesurer l’état de  forme d’un joueur à un moment donné. Le principe est fondé sur le coefficient elo établi  pour les joueurs d’échecs. Ici, chaque joueur débute avec une valeur elo de 1500 points lors de son  premier match, et gagne ou perd des points après chaque victoire ou défaite. La valeur elo  fluctuant dans le temps en fonction de l’accumulation de victoires et de défaites, elle est une  indication de l’état de forme d’un joueur à une date donnée.  
       
        """
    )
    st.markdown(
     """
     :tennis: Nous remarquons qu'il y a des données manquantes dans le jeu de données  mais elles se trouvent dans  les colonnes relatives aux cotes des bookmakers, ces cotes étant indisponibles pour les tournois les plus anciens...

     """
     )

    #st.image(Image.open("assets/sample-image.jpg"))
    
    
    st.markdown(
        """
        :point_right: [La Data Visualisation](https://app.powerbi.com/links/riU_8qHnWI?ctid=e8df5f43-80c9-47f6-a040-4562dce73ccd&pbi_source=linkShare)
            
        """
    )
    st.markdown(
        """
        ## Mais dans tout cela les bookmakers :moneybag: :moneybag: leurs prédictions sont-elles justes ?
       
        """
    )
    st.markdown(
        """
        Nous avons deux bookmakers dans le dataframe PS et B365 mais après comparaison des moyennes et médianes de leurs cotes nous avons constaté que leurs prédictions étaient quasi équivalentes. Donc pour simplifier notre analyse nous avons gardé un seul bookmaker le B365 car il a  le plus grand nombre de cotes.
        
        
        """
    )
    st.markdown(
         """
         
         Mais d'ailleurs la **cote** c'est quoi ? Elle a deux rôles, elle permet de :
         - Connaitre la prédiction du bookmaker entre deux joueurs car la cote la plus petite représentera le favori du bookmaker
         - Connaitre aussi son gain ( exemple cote 1.50 sur le joueur A en cas de victoire de celui-ci le gain sera de 1.50 * mise de départ )
         
         """
     )
    st.markdown(
          """
          Intéressons nous aux prédictions de notre bookmaker (B365) avec une représentation graphique de ses prédictions(après suppressions des données manquantes).
          
          """
      )
    
    @st.cache(suppress_st_warning=True,allow_output_mutation=True)
    def predict(df):
        data = pd.DataFrame()
        data = df
        data = data.dropna()

        # Synthèse des prévisions des bookmakers dans un dataframe 
        data['Bkm_prediction'] = data[['B365W','B365L']].apply(lambda x: np.argmin(x), axis=1)
        data['Bkm_prediction'] = data['Bkm_prediction'].replace(to_replace=[0, 1], value=['V', 'D'])

        # Transformer les valeurs de la variable Winner en "V" comme victoire pour comparer les prév et le réel
        data['Victoire_reel'] = "V"
        data['Predict_bkm'] = data[['Bkm_prediction']] 
        data["Bkm_predict_vict"] = data["Predict_bkm"].replace({"D":0,"V":1}).astype(float)

        # Le pourcentage de bonnes prédictions
        data['Bkm_prediction'] = data['Victoire_reel']==data['Bkm_prediction']
        label = data['Bkm_prediction'].value_counts().index
        fig = plt.figure(figsize=(2,2))
        plt.pie(x=data['Bkm_prediction'].value_counts().values, 
        autopct="%.1f%%", 
        labels=label,
        explode = [ 0, 0.2], 
        pctdistance=0.5,
        shadow = True)
        plt.title('Pourcentage de bonnes prédictions des bookmakers');
        st.pyplot(fig)
     

        # Diviser le dataframe en deux en ayant un joueur par ligne en créant une colonne Win en mettant 0 si le joueur est perdant et 1 si il est gagnant 
        data['RankDiff'] = data.LRank - data.WRank

        winners = pd.DataFrame(data = [data.Winner, data.Location, data.Tournament, data.Date, data["Best of"], data.Series, data.Court, data.Surface, data.Round, data.WRank, data.Wsets, data.elo_winner,data.B365W, data.RankDiff,data["Bkm_predict_vict"]]).T
        winners.columns =['Player', 'Location', 'Tournament', 'Year', 'BestOf', 'Series', 'Court', 'Surface','Round', 'Rank', 'SetsWon', 'EloPoints', 'B365','RankDiff','Predict_W_Bkm']
        winners['Win'] = 1
        losers = pd.DataFrame(data = [data.Loser, data.Location, data.Tournament, data.Date, data["Best of"], data.Series, data.Court, data.Surface, data.Round, data.LRank, data.Lsets, data.elo_loser,data.B365L, data.RankDiff, data["Bkm_predict_vict"]]).T
        losers.columns =['Player', 'Location', 'Tournament', 'Year', 'BestOf', 'Series', 'Court', 'Surface', 
        'Round', 'Rank', 'SetsWon', 'EloPoints','B365', 'RankDiff','Predict_W_Bkm']
        losers['Win'] = 0
        new_df = pd.concat([winners, losers], axis = 0)
        return new_df
    new_df = predict(data)
    new_df['Year'] = pd.to_datetime(new_df['Year'])
    new_df_strategie = new_df.sort_values(by=["Year"],ascending = True)
    new_df = new_df_strategie.drop('Predict_W_Bkm',axis = 1)
    
    
    st.markdown(
        """
        Le pourcentage est de 71.4 % donc **l'objectif du projet est d'être au delà de 72 % de bonnes prévisions...**
        
        """
    )
    
    st.markdown(
        """
        Avant d'attaquer la partie préprocessing nous avons enrichi notre jeu de données avec des moyennes roulantes c'est à dire calculer le ratio de victoire sur les 6 derniers mois et les 18 derniers mois en filtrant sur le nombre total de victoires mais en fonction de la surface , du tournoi et du tour.
        Nous avons donc deux indicateurs solides pour entrainer notre modèle ( elo et les moyennes roulantes )
        
        """
    )
    st.markdown(
        """
        ## Statistiques Joueurs :
       
        """
      )
    
    
    
    # Moyenne roulante stat joueurs
    @st.cache(suppress_st_warning=True,allow_output_mutation=True)
    def mean_rolling(new_df,x,y):
      # Moyenne roulante des victoires par joueur sur x et y mois en fonction des victoires totales, de la surface, du tournois et des tours 
      new_df['Year'] = pd.to_datetime(new_df['Year'])
      var_mois = [ x , y ]
    
      for i in var_mois :
        new_df = new_df.sort_values(by=["Player","Year"],ascending = True)
        new_df[f"Ratio_victoire_{i}_mois"] = ((new_df.groupby("Player").rolling(str(i*30) + "D", min_periods=1, on="Year", closed="both")["Win"].apply(lambda x: x.shift(1).sum() / (len(x) - 1))).fillna(0).values)
        new_df[f"Ratio_surface_{i}_mois"] = ((new_df.groupby(["Player","Surface"]).rolling(str(i*30) + "D", min_periods=1, on="Year", closed="both")["Win"].apply(lambda x: x.shift(1).sum() / (len(x) - 1))).fillna(0).values)
        new_df[f"Ratio_tournois_{i}_mois"] = ((new_df.groupby(["Player","Tournament"]).rolling(str(i*30) + "D", min_periods=1, on="Year", closed="both")["Win"].apply(lambda x: x.shift(1).sum() / (len(x) - 1))).fillna(0).values)
        new_df[f"Ratio_tours_{i}_mois"] = ((new_df.groupby(["Player","Round"]).rolling(str(i*30) + "D", min_periods=1, on="Year", closed="both")["Win"].apply(lambda x: x.shift(1).sum() / (len(x) - 1))).fillna(0).values)
      return new_df
  
    # 6 et 18 seront les mois choisis pour la fonction
    new_df = mean_rolling(new_df,6,18)
    new_df = new_df.sort_values(by=["Year"],ascending = True).reset_index(drop= True)
    new_df["Year"] = pd.to_datetime(new_df["Year"])
    new_df["Year"]= new_df["Year"].dt.date
    st.write(new_df)
    new_df.to_csv('df_variables_enrichies.csv',index = False)
    new_df["Year"] = pd.to_datetime(new_df["Year"])
    new_df["Year"]= new_df["Year"].dt.date
    players = new_df['Player'].unique()
    players_choice = st.selectbox('Sélectionner un joueur:', players,key=20)
    years = new_df["Year"].loc[new_df["Player"] == players_choice]
    year_choice = st.selectbox('Sélectionner une date', years) 
    st.write('Resultat de la recherche:',new_df.loc[(new_df["Player"] == players_choice)&(new_df["Year"] == year_choice)])
      

    
    
    
   