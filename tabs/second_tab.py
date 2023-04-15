"""
Created on Sun Oct 16 16:16:13 2022.

@author: olecu
"""




# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    def load_data():
        data = pd.read_csv('atp_data.csv', parse_dates=['Date'])
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
        :point_right: [La Data Visualisation](https://app.powerbi.com/links/riU_8qHnWI?ctid=e8df5f43-80c9-47f6-a040-4562dce73ccd&pbi_source=linkShare) avec Power Bi
    
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
        # Supprimer les lignes avec des valeurs manquantes
        data = df.dropna()
    
        # Calculer la prédiction du bookmaker B365 (0 pour la défaite et 1 pour la victoire)
        data['Bkm_prediction'] = np.argmin(data[['B365W', 'B365L']], axis=1)
        data['Bkm_prediction'] = data['Bkm_prediction'].replace(to_replace=[0, 1], value=['D', 'V'])
    
        # Transformer les valeurs de la variable Winner en "V" comme victoire pour comparer les prév et le réel
        data['Victoire_reel'] = "V"
    
        # Calculer le pourcentage de bonnes prédictions des bookmakers
        bkm_accuracy = (data['Victoire_reel'] == data['Bkm_prediction']).value_counts(normalize=True)[True] * 100
        st.markdown(f"Le pourcentage de bonnes prédictions des bookmakers est de : {bkm_accuracy:.2f}%")
    
        # Diviser le dataframe en deux en ayant un joueur par ligne en créant une colonne Win en mettant 0 si le joueur est perdant et 1 si il est gagnant
        winners = data[['Winner', 'Location', 'Tournament', 'Date', 'Best of', 'Series', 'Court', 'Surface', 'Round', 'WRank', 'Wsets', 'elo_winner', 'B365W', 'LRank', 'Bkm_prediction']]
        winners.columns = ['Player', 'Location', 'Tournament', 'Year', 'BestOf', 'Series', 'Court', 'Surface', 'Round', 'Rank', 'SetsWon', 'EloPoints', 'B365', 'RankDiff', 'Predict_W_Bkm']
        winners['Win'] = 1
    
        losers = data[['Loser', 'Location', 'Tournament', 'Date', 'Best of', 'Series', 'Court', 'Surface', 'Round', 'LRank', 'Lsets', 'elo_loser', 'B365L', 'WRank', 'Bkm_prediction']]
        losers.columns = ['Player', 'Location', 'Tournament', 'Year', 'BestOf', 'Series', 'Court', 'Surface', 'Round', 'Rank', 'SetsWon', 'EloPoints', 'B365', 'RankDiff', 'Predict_W_Bkm']
        losers['Win'] = 0
    
        new_df = pd.concat([winners, losers], axis=0, ignore_index=True)
    
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
        
    def mean_rolling(df, x, y):
        """
        Cette fonction calcule la moyenne roulante des victoires par joueur sur x et y mois en fonction des victoires totales,
        de la surface, du tournoi et des tours.
        
        :param df: le dataframe de données
        :param x: la période de temps en mois pour la première moyenne roulante
        :param y: la période de temps en mois pour la deuxième moyenne roulante
        :return: le dataframe de données avec deux colonnes supplémentaires pour chaque moyenne roulante calculée
        """
        df['Year'] = pd.to_datetime(df['Year'])
        var_mois = [x, y]
    
        for i in var_mois:
            df = df.sort_values(by=['Player', 'Year'], ascending=True)
            col_name = f'Ratio_victoire_{i}_mois'
            df[col_name] = (
                df.groupby('Player')['Win']
                .apply(lambda x: x.rolling(f'{i}M').sum().shift().fillna(0) / (x.rolling(f'{i}M').count() - 1))
                .values
            )
            col_name = f'Ratio_surface_{i}_mois'
            df[col_name] = (
                df.groupby(['Player', 'Surface'])['Win']
                .apply(lambda x: x.rolling(f'{i}M').sum().shift().fillna(0) / (x.rolling(f'{i}M').count() - 1))
                .values
            )
            col_name = f'Ratio_tournois_{i}_mois'
            df[col_name] = (
                df.groupby(['Player', 'Tournament'])['Win']
                .apply(lambda x: x.rolling(f'{i}M').sum().shift().fillna(0) / (x.rolling(f'{i}M').count() - 1))
                .values
            )
            col_name = f'Ratio_tours_{i}_mois'
            df[col_name] = (
                df.groupby(['Player', 'Round'])['Win']
                .apply(lambda x: x.rolling(f'{i}M').sum().shift().fillna(0) / (x.rolling(f'{i}M').count() - 1))
                .values
            )
    
        return df
    
    
