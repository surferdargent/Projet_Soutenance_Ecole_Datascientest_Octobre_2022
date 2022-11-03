import streamlit as st




#from traitement import y_test,X_test,y_train,split_normalisation,mean_rolling,creat_df




title = "Battre les bookmakers Tennis ?"
header = ("C'est possible !!!")
sidebar_name = "Introduction"


def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
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
        Le jeu de données utilisé est sur [kaggle](https://www.kaggle.com/edouardthomas/atp-matches-dataset) il représente tous les matchs de tennis entre 2000 et 2018.
        
        """
    )






    