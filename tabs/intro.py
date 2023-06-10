"""

Created on Sun Oct 16 16:16:13 2022.

@author: olecu
"""
import streamlit as st

title = "Battre les bookmakers Tennis ?"
header = ("C'est possible !!!")
sidebar_name = "Introduction"


def run():
    """Voir prédiction des bookmakers."""
    st.image("https://cdn.dribbble.com/users/891352/screenshots/3473217/media/db7d2f04eab893dda0a2afec79e2311f.gif",
             width=450)
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")
    st.title(title)
    st.header(header)
    st.markdown("---")
    st.markdown(
        """
        L’objectif de ce projet est :
        - De battre les algorithmes des bookmakers sur l’estimation de la probabilité d’un joueur de gagner un match
        - De dégager une stratégie de paris
        Le jeu de données utilisé est sur [kaggle](https://www.kaggle.com/edouardthomas/atp-matches-dataset) il représente tous les matchs de tennis entre 2000 et 2018.""")
if __name__ == "__main__":
    run()