"""
Created on Sun Oct 16 16:16:13 2022.

@author: olecu
"""

# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
from traitement import y_pred_rf, probs

title = "Stratégie et Conclusion"
sidebar_name = "Stratégie et Conclusion"


def run():
    """Lancement de la page 4."""
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
    data = pd.read_csv('atp_data.csv', parse_dates=['Date'])
    data["Date"] = pd.to_datetime(data["Date"])
    data['Date'] = data['Date'].dt.date
    data.rename(columns={'Date': 'Year'}, inplace=True)

    # Synthèse des prévisions des bookmakers dans un dataframe
    Bkm_prediction = data[['B365W', 'B365L']].apply(lambda x: np.argmin(x), axis=1)
    data['Bkm_prediction'] = Bkm_prediction
    data['Bkm_prediction'] = data['Bkm_prediction'].replace(to_replace=[0, 1], value=['V', 'D'])


# Transformer les valeurs de la variable Winner en "V" comme victoire pour comparer les prév et le réel
    data['Victoire_reel'] = "V"
# data[['Bkm_prediction','Victoire_reel','Winner']]
    data["Bkm_predict_vict"] = data['Bkm_prediction'].replace({"D": 0, "V": 1}).astype(float)

    col1, col2 = st.columns([1, 1])

    with col1:
        mise_de_depart = st.number_input('Mise de départ', 1, 100, 10, key=2)
        start_date, end_date = st.date_input('Choisir date de début, date de fin:', [datetime(2016, 1, 1), datetime(2018, 3, 4)], key=6)

        if start_date < end_date:
            pass
        # else:
        #     st.error('Error: Date de fin doit être choisi après la date de début.')

        # greater than the start date and smaller than the end date

        mask = (data['Year'] > start_date) & (data['Year'] <= end_date)
        data = data.loc[mask]

    with col2:

        st.image(Image.open("assets/Argent2.png"))

    # st.dataframe(data)
    # # mask = (data['Year'] > start_date) & (data['Year'] <= end_date)
    # # data = data.loc[mask]

    # st.dataframe(data)

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

    data["Mise"] = data['Bkm_predict_vict'] * mise_de_depart
    data["Gain"] = data["Mise"] * (data["B365W"]-1)

    st.write("La somme pariée serait de", round(data["Mise"].sum(axis=0), 2), "euros et le gain prédit de", round(data["Gain"].sum(axis=0), 2), "euros.Soit", round(round(data["Gain"].sum(axis=0), 2)/round(data["Mise"].sum(axis=0)), 2)*100, "% de bénéfices.Ce gain représente la somme gagnée si nous suivons les préco du bookmaker B365")

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

    data_var = pd.read_csv('df_variables_enrichies.csv', parse_dates=['Year'])
    date_split = pd.Timestamp(2016, 1, 1)
    data_var = data_var[data_var['Year'] >= date_split]
    data_var['Year'] = pd.to_datetime(data_var['Year'])
    data_var['Year'] = data_var['Year'].dt.date

    # data.reset_index(drop=True, inplace=True)
    data_var = data_var.sort_values(by=["Year"], ascending=True)
    data_var.reset_index(inplace=True, drop=False)
    data_var["Prédictions_Algo"] = y_pred_rf
    first_column = data_var.pop("Prédictions_Algo")

    # insert column using insert(position,column_name,first_column) function
    data_var.insert(0, "Prédictions_Algo", first_column)

    start_date_var, end_date_var = start_date, end_date

    if start_date_var < end_date_var:
        pass

    else:
        st.error('Erreur: Date de fin doit être choisi après la date de début.')

    mask_var = (data_var['Year'] >= start_date_var) & (data_var['Year'] <= end_date_var)
    data_var_mask = data_var.loc[mask_var]

    def paris1(mise_de_dep, data_var_mask):
        mise_de_depart = mise_de_dep
        data_var_mask["Gains"] = data_var_mask["Prédictions_Algo"] * mise_de_depart * (data_var_mask['B365']-1)
        data_var_mask["Mise"] = data_var_mask["Prédictions_Algo"] * mise_de_depart
        second_column = data_var_mask.pop("Mise")
        data_var_mask.insert(1, "Mise", second_column)
        third_column = data_var_mask.pop("Gains")
        data_var_mask.insert(2, "Gains", third_column)
        st.dataframe(data_var_mask)
        return data_var_mask

    paris1(mise_de_depart, data_var_mask)
    st.write("La somme pariée serait de 116070 euros et le gain prédit de 49807 euros si nous suivons les recommandations des bookmakers sur notre jeu de test .Soit 43.0 % de bénéfices")

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
       """)

    data_var_mask['Prob de perdre'] = probs[:, 0]
    data_var_mask['Prob de gagner'] = probs[:, 1]
    fourth_column = data_var_mask.pop("Prob de perdre")
    data_var_mask.insert(3, "Prob de perdre", fourth_column)

    five_column = data_var_mask.pop("Prob de gagner")
    data_var_mask.insert(4, "Prob de gagner", five_column)

    def paris2(seuil, data_var_mask):

        seuil_pari = seuil
        data_var_mask_seuil = data_var_mask[data_var_mask["Prob de gagner"] >= seuil_pari]

        st.dataframe(data_var_mask_seuil)
        return data_var_mask_seuil
    paris2(0.8, data_var_mask)

    # st.write("La somme pariée serait de", round(data_var_mask_seuil["Mise"].sum(),2), "euros et le gain prédit de", round(data_var_mask_seuil["Gains"].sum(),2),"euros."),st.write("Soit",round( (data_var_mask_seuil["Gains"].sum()-data_var_mask_seuil["Mise"].sum())/data_var_mask_seuil["Mise"].sum(),2)*100,"% de bénéfices")

    st.markdown(
        """

        La somme pariée serait de 20600 euros et le gain prédit de 7843 euros.Soit 38.0 % de bénéfices.


         La dernière stratégie serait la plus optimale car le bénéfice est de 38,0 % légèrement supérieur au bénéfice des préconisations bookmakers ( 36,0 %)
         Surtout la somme engagée pour la dernière stratégie est de 20600 euros contre 41120 euros si on suit les préconisations bookmakers.Notre stratégie battrait les bookmakers ...""")

    # Pour le test nous allons prendre les 10 derniers joueurs de notre jeu de test et appliquer notre  stratégie

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
        par rapport à une situation dans laquelle nous n’aurions pas été en mesure de prédire avec une confiance suffisante.
        Avec davantage de temps et forts de l’expérience acquise lors du projet, il nous apparaît aujourd’hui que nous aurions pu explorer davantage la piste suivante.Lors de la phase exploratoire des données
        nous avons produit deux types de statistiques qui auraient pu éventuellement servir à élaborer une stratégie de paris plus efficace.
        Nous aurions ainsi pu établir d’une part des statistiques de performance des joueurs selon d’autres paramètres (surface, tournoi, adversaire...), et d’autre part identifier les types de matchs pour lesquels les pronostics des bookmakers sont moins performants. Cela aurait
        pu nous permettre de mener des stratégies de niche exploitant les défaillances des
        bookmakers.
    """
    )

if __name__ == "__main__":
    run()