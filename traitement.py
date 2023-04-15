# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 16:16:13 2022

@author: olecu
"""

# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import io   
import config 
import pickle

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression 
from PIL import Image
from datetime import datetime 
from sklearn.neighbors import KNeighborsClassifier


sns.set_theme()  






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


# Suppression des manquants
data = data.dropna()

# Synthèse des prévisions des bookmakers dans un dataframe 

data['Bkm_prediction'] = data[['B365W','B365L']].apply(lambda x: np.argmin(x), axis=1)
data['Bkm_prediction'] = data['Bkm_prediction'].replace(to_replace=[0, 1], value=['V', 'D'])


# Transformer les valeurs de la variable Winner en "V" comme victoire pour comparer les prév et le réel
data['Victoire_reel'] = "V"
#data[['Bkm_prediction','Victoire_reel','Winner']]
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
 
 
 # Diviser ke dataframe en deux en ayant un joueur par ligne en créant une colonne Win en mettant 0 si le joueur est perdant et 1 si il est gagnant 
data['RankDiff'] = data.LRank - data.WRank

winners = pd.DataFrame(data = [data.Winner, data.Location, data.Tournament, data.Date, data["Best of"], data.Series, data.Court, data.Surface, data.Round, data.WRank, data.Wsets, data.elo_winner,data.B365W, data.RankDiff,data["Bkm_predict_vict"]]).T

winners.columns =['Player', 'Location', 'Tournament', 'Year', 'BestOf', 'Series', 'Court', 'Surface','Round', 'Rank', 'SetsWon', 'EloPoints', 'B365','RankDiff','Predict_W_Bkm']

winners['Win'] = 1

losers = pd.DataFrame(data = [data.Loser, data.Location, data.Tournament, data.Date, data["Best of"], data.Series, data.Court, data.Surface, data.Round, data.LRank, data.Lsets, data.elo_loser,data.B365L, data.RankDiff, data["Bkm_predict_vict"]]).T

losers.columns =['Player', 'Location', 'Tournament', 'Year', 'BestOf', 'Series', 'Court', 'Surface', 
               'Round', 'Rank', 'SetsWon', 'EloPoints','B365', 'RankDiff','Predict_W_Bkm']

losers['Win'] = 0


new_df = pd.concat([winners, losers], axis = 0)

new_df['Year'] = pd.to_datetime(new_df['Year'])
new_df_strategie = new_df.sort_values(by=["Year"],ascending = True)
new_df = new_df_strategie.drop('Predict_W_Bkm',axis = 1)

 # Moyenne roulante stat joueurs

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
players_choice = st.selectbox('Sélectionner un joueur:', players)
years = new_df["Year"].loc[new_df["Player"] == players_choice]
year_choice = st.selectbox('Sélectionner une date', years) 
st.write('Resultat de la recherche:',new_df.loc[(new_df["Player"] == players_choice)&(new_df["Year"] == year_choice)])



 
st.title("Préprocessing et Modélisation")

st.markdown(
 """
   ## Préprocessing
 """
 )
 
st.markdown(
 """
   :tennis: Les variables :
 """
 )
st.markdown(
 """
 Il faut avant de s'attaquer à la modélisation encoder nos variables.
 
 """
 )

new_df_preprocessing = pd.read_csv('df_variables_enrichies.csv',parse_dates=['Year'])
new_df_preprocessing['Year'] = pd.to_datetime(new_df_preprocessing['Year'])
new_df_preprocessing['Year'] = new_df_preprocessing['Year'].dt.date
 # Affichage info df df
 # st.write(new_df_preprocessing)
 
col1,col2= st.columns([3,1])

with col1:
    st.markdown("""*Nos variables sont elles bien typées ?*""")
    buffer = io.StringIO()
    new_df_preprocessing.info(buf=buffer)

    s = buffer.getvalue()
    st.text(s)
with col2:
    st.markdown("""
 
            _*Action*_
            """
            )
st.markdown(
 
    """
     Toutes les variables de type "object" devront être encodées ..."""
)
 
st.markdown(
 """
 :tennis: Pour l'encodage nous avons choisi la méthodologie suivante :
 - Pour les variables **Player**, **Tournament**, **Series** et **Round** nous remplaçons les noms par des numéros id.
 - Pour la variable **Court** nous remplacons ‘Outdoor’, ‘Indoor’ par 0, 1 et pour la variable **Surface** nous changeons les modes 'Hard', 'Clay’, 'Grass', 'Carpet' par 0, 1, 2, 3.
 - Pour les variables numériques mais qui sont au format texte nous les typons en float.

 """
 )
 
 # Encodage des variables 
 # Certaines variables sont catégorielles il faut les passer en numérique
 
 # Player
Id_player= pd.DataFrame(new_df_preprocessing['Player'].unique(), columns=['Name'])
Id_player = Id_player.rename_axis('Id_player').reset_index()
new_df_preprocessing = pd.merge(new_df_preprocessing, Id_player,left_on="Player", right_on="Name")
new_df_preprocessing["Id_player"]= new_df_preprocessing['Id_player'].astype(float)
 
 # Tournament 
 
Id_tournament= pd.DataFrame(new_df_preprocessing['Tournament'].unique(), columns=['Tourna'])
Id_tournaments = Id_tournament.rename_axis('Id_tournament').reset_index()
new_df_preprocessing = pd.merge(new_df_preprocessing, Id_tournaments,left_on="Tournament", right_on="Tourna")
new_df_preprocessing["Id_tournament"]= new_df_preprocessing['Id_tournament'].astype(float)
 
 # Series
 
Id_Series= pd.DataFrame(new_df_preprocessing['Series'].unique(), columns=['Type_series'])
Id_Series = Id_Series.rename_axis('Id_series').reset_index()
new_df_preprocessing = pd.merge(new_df_preprocessing, Id_Series,left_on="Series", right_on='Type_series')
new_df_preprocessing["Id_series"]= new_df_preprocessing["Id_series"].astype(float)
 
 # Surface
new_df_preprocessing[ 'Surface' ] = new_df_preprocessing[ 'Surface' ].replace( to_replace =[ 'Hard' , 'Clay' , 'Grass','Carpet' ], value =[0,1,2,3])
new_df_preprocessing[ 'Surface' ].astype(float)
 
 # Court
new_df_preprocessing[ 'Court' ] = new_df_preprocessing[ 'Court' ].replace( to_replace =[ 'Outdoor' , 'Indoor' ], value =[0,1])
new_df_preprocessing[ 'Court' ].astype(float)
 
 # Round
 
Id_round= pd.DataFrame(new_df_preprocessing['Round'].unique(), columns=['Type_round'])
Id_round = Id_round.rename_axis('Id_round').reset_index()
new_df_preprocessing = pd.merge(new_df_preprocessing, Id_round,left_on="Round", right_on='Type_round')
new_df_preprocessing["Id_round"]= new_df_preprocessing["Id_round"].astype(float)
 
new_df_preprocessing= new_df_preprocessing.drop(['Player', 'Location', 'Tournament',  'Series', 
     'Round','Name', 'Tourna',
    'Type_series',  'Type_round'],axis=1)
 
 # Passer les variables object numériques en float 
 
new_df_preprocessing[['BestOf', 'Court', 'Surface', 'Rank', 'SetsWon', 'EloPoints',
'B365', 'RankDiff', 'Win', 'Id_player', 'Id_tournament', 'Id_series',
'Id_round']]=new_df_preprocessing[['BestOf', 'Court', 'Surface', 'Rank', 'SetsWon', 'EloPoints',
'B365', 'RankDiff', 'Win', 'Id_player', 'Id_tournament', 'Id_series',
'Id_round']].astype(float)
 
st.markdown(
 """
 
 Comme le montre le dataframe ci-desous les variables sont bien toutes au format numérique nous pouvons définir nos features :chart_with_upwards_trend:, notre target :dart: et normaliser nos variables .

 """
 )
st.write(new_df_preprocessing.head())
 
st.markdown(
 """
  
 :tennis: La variable **"Win"** sera notre target.
 
 
 """
 )
 
_left, mid, _right = st.columns(3)
with _left:
    st.image("./assets/Diapositive3.PNG",width=650)

from sklearn.preprocessing import StandardScaler
 
# Modèle de classification que l'on va utiliser

from sklearn.ensemble import RandomForestClassifier


 # Trier les dates du dataset 
new_df_preprocessing = new_df_preprocessing.sort_values(by=["Year"],ascending = True)
 # @st.cache(suppress_st_warning=True)
 # @st.cache(allow_output_mutation=True)
def split(data):
    df = pd.DataFrame(data)
    df = data.sort_values(by=["Year"],ascending = True)
    
    # Diviser le dataset en "train" et "test" toutes les données avant le 01 janvier 2016 seront égales au "train" et après au test
    date_split = pd.Timestamp(2016, 1, 1)
    data_train = df[df['Year'] < date_split]
    data_test =  df[df['Year'] >= date_split]
    
    # Création des quatres variables pour l'entrainement et le test ( X_train, X_test, y_train, y_test )
    X_train = data_train.drop(['Win'], axis=1)
    X_test =  data_test.drop(['Win'], axis=1)
    
    y_train = data_train['Win']
    y_test =  data_test['Win']
    
    
    X_train = X_train.select_dtypes('float')
    X_test = X_test.select_dtypes('float')
    
    
    # On normalise nos données numériques :
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
    
    
    X_train = X_train_scaled
    X_test = X_test_scaled
    
    #y_test = y_test.reset_index(drop=True)
    return X_train,y_train,X_test,y_test
st.markdown("""Nous pouvons passer à la modélisation.""")
st.markdown("---")
st.markdown(
"""
  ## Modélisation
"""
)
 
st.markdown(""":tennis: 1er entraînement""")

X_train,y_train,X_test,y_test = split(new_df_preprocessing) 

new_y_test = pd.Series(y_test,index=None)

#new_y_test=y_test  


# Définition du modèle

  
 


 # Exécution des modèles
 # @st.cache(suppress_st_warning=True)
 # @st.cache(allow_output_mutation=True)
def train_model():

    models = []
    models.append(('Logistic Regression',LogisticRegression(random_state=123)))
    models.append(('KNeighbors', KNeighborsClassifier()))
    models.append(('Random Forest', RandomForestClassifier(random_state=123)))
    accuracies = []
    names = []
 
    for name, model in models:
        model.fit(X_train,y_train)
        accuracy = model.score(X_test,y_test)
        accuracies.append(accuracy)
        names.append(name)
        msg = "Résultat pour %s: %f" % (name, accuracy)
        st.write(msg)
    fig = plt.figure()
    sns.barplot(names, accuracies)
    plt.show()
    st.pyplot(fig)      
train_model()
 
st.markdown(
"""
Après avoir normalisé et entrainé les modèles, nous avons obtenu des scores qui sont au dessus de 95 %.
Les performances de nos modèles laissent penser qu’il y a sans doute un surapprentissage, qu’il nous faut identifier. 
Pour ce faire, nous avons établi une *matrice de  corrélation* :
"""
)
cor = new_df_preprocessing.corr()
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(cor,ax=ax, cmap='coolwarm');
st.write(fig)
 
st.markdown("""
 Nous constatons que la variable "SetsWon" est fortement corrélée avec "Win" ce qui est normal car elle donne une indication sur le nombre de sets gagnés sur le match.
 Par contre cette information nous ne l'avons pas avant le match il faut donc la supprimer des features et relancer le modèle.
 
 """)
 
st.markdown(""":tennis: 2ème entraînement """)
 
new_df_preprocessing_demo = new_df_preprocessing.drop("SetsWon",axis=1)
X_train,y_train,X_test,y_test = split(new_df_preprocessing_demo) 

 
train_model()
st.markdown("""
 Ces résultats semblent plus conformes à ce que l’on peut attendre pour ce type de données.  
 Le meilleur modèle est RF avec un score de 0.88, ce qui est meilleur que les prédictions des bookmakers. 
 Nous allons maintenant tenter d’améliorer les performances du modèle. 
 """)
 
  
 # Fonction split et normalisation des données
 # @st.cache(suppress_st_warning=True)
 # @st.cache(allow_output_mutation=True)
def split_normalisation(data,option):
    
  df = pd.DataFrame(data)
  df = data.sort_values(by=["Year"],ascending = True)
  x = option
   
  

# Diviser le dataset en "train" et "test" toutes les données avant le 01 janvier 2016 seront égales au "train" et après au test
  date_split = pd.Timestamp(2016, 1, 1)
  data_train = df[df['Year'] < date_split]
  data_test =  df[df['Year'] >= date_split]

 # Création des quatres variables pour l'entrainement et le test ( X_train, X_test, y_train, y_test )
  X_train = data_train.drop(['Win'], axis=1)
  X_test =  data_test.drop(['Win'], axis=1)
  y_train = data_train['Win']
  y_test =  data_test['Win']


  X_train = X_train.select_dtypes('float')
  X_test = X_test.select_dtypes('float')


# On normalise nos données numériques :
  scaler = StandardScaler()
  X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
  X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
  X_train = X_train_scaled
  X_test = X_test_scaled
  models = []
  models.append(('Logistic Regression',LogisticRegression(random_state=123)))
  models.append(('KNeighbors', KNeighborsClassifier()))
  models.append(('Random Forest', RandomForestClassifier(random_state=123)))
  accuracies = []
  names = []
  data=[]
    
  for name, model in models:
        model.fit(X_train,y_train)
        accuracy = model.score(X_test,y_test)
        accuracies.append(accuracy)
        names.append(name)
        data.append([names,accuracies]) 
        # with open(f".\models\{name}.sav",'wb') as f:
        #   pickle.dump(model,f)   
        df = pd.DataFrame(list(zip(names,accuracies)), columns=['Noms', f"Scores {x}"])
      
  return df
 
def importance_variables():
    fig1 = plt.figure(figsize=(14,6))
    train_features = X_train
    rf_loaded = pickle.load(open('.\models\Random Forest.sav', 'rb'))
    vars_imp = pd.Series(rf_loaded.feature_importances_,index=train_features.columns).sort_values(ascending=False)
    sns.barplot(x=vars_imp.index,y=vars_imp)
    plt.xticks(rotation=90)
    plt.xlabel('Variables')
    plt.ylabel("Scores d'importance de la variable")

    plt.show()
    st.write("""Ci dessous l'importance des variables nous donne des indications sur le poids de chaque variable sur notre modèle.
    """)  
    return st.pyplot(fig1)
importance_variables()
st.markdown("""
 :tennis: Amélioration du modèle RF
 
 Nous allons procéder en deux étapes: nous allons tout d’abord identifier les paramètres qui  apportent le meilleur gain de performances en changeant les variables d’entraînement, puis  nous réglerons ensuite les hyperparamètres. 
 """)

st.markdown("""
 - Analyses de la performance des variables
 
 Une première approche consiste à utiliser de façon sélective les différents paramètres.  


 """)

#Features d'origine à conserver
drop_variable1 = ['EloPoints','RankDiff', 'Ratio_tours_6_mois','Ratio_victoire_6_mois', 'Ratio_surface_6_mois', 'Ratio_tournois_6_mois' , 'Ratio_tournois_18_mois', 'Ratio_tours_18_mois','Ratio_victoire_18_mois', 'Ratio_surface_18_mois']
df1 = new_df_preprocessing_demo.drop(drop_variable1,axis=1)
df1 = split_normalisation(df1,1)
 

# "Features d'origine + Points ELO"  à conserver
drop_variable2 = ['RankDiff',
'Ratio_tours_6_mois','Ratio_victoire_6_mois', 'Ratio_surface_6_mois', 'Ratio_tournois_6_mois' ,
'Ratio_tournois_18_mois', 'Ratio_tours_18_mois','Ratio_victoire_18_mois', 'Ratio_surface_18_mois']
df2 = new_df_preprocessing_demo.drop(drop_variable2,axis=1)
df2 = split_normalisation(df2,2)

 
#"Features d'origine + Points ELO + Diff. de classement "  à conserver
drop_variable3 = ['Ratio_tours_6_mois','Ratio_victoire_6_mois', 'Ratio_surface_6_mois', 'Ratio_tournois_6_mois' ,
'Ratio_tournois_18_mois', 'Ratio_tours_18_mois','Ratio_victoire_18_mois', 'Ratio_surface_18_mois']
df3 = new_df_preprocessing_demo.drop(drop_variable3,axis=1)
df3 = split_normalisation(df3,3)

 
#"Features d'origine + Points ELO + Diff. de classement + Moy.roulantes 6 mois"  à conserver
drop_variable4 = ['Ratio_tournois_18_mois', 'Ratio_tours_18_mois','Ratio_victoire_18_mois', 'Ratio_surface_18_mois']
df4 = new_df_preprocessing_demo.drop(drop_variable4,axis=1)
df4 = split_normalisation(df4,4)
 

#"Features d'origine + Points ELO + Diff. de classement + Moy.roulantes 6 mois + Moy.roulantes 18 mois"  à conserver
df5 = new_df_preprocessing_demo  
df5 = split_normalisation(df5,5)
 


# model_choisi = st.selectbox(label = "Choix des Features", options = ["Features d'origine ", "Features d'origine + Pts ELO", "Features d'origine + Pts ELO + Diff. de classement","Features d'origine + Pts ELO + Diff. de classement + Moy.roulantes 6 mois","Features d'origine + Pts ELO + Diff. de classement + Moy.roulantes 6 mois + Moy.roulantes 18 mois"])
st.markdown("""
    Bilan des performances des différentes options
           """)

df6 = df1
df1 = df1["Scores 1"]
df2 = df2["Scores 2"]
df3 = df3["Scores 3"]
df4 = df4["Scores 4"]
df5 = df5["Scores 5"]
df6=df6["Noms"]
@st.cache(allow_output_mutation=True)
def creat_df():
    
    data1 =  pd.concat([df6,df1],axis=1)
     
    data2 =  pd.concat([df6,df1,df2],axis=1)
     
    data3 =  pd.concat([df6,df1,df2,df3],axis=1)
     
    data4 =  pd.concat([df6,df1,df2,df3,df4],axis=1)
     
    data5 =  pd.concat([df6,df1,df2,df3,df4,df5],axis=1)
    return (data1,data2,data3,data4,data5)
data1,data2,data3,data4,data5 = creat_df()


selections = ["Features d'origine" , "Features d'origine + Pts ELO", "Features d'origine + Pts ELO + Diff. de classement","Features d'origine + Pts ELO + Diff. de classement + Moy.roulantes 6 mois","Features d'origine + Pts ELO + Diff. de classement + Moy.roulantes 6 mois + Moy.roulantes 18 mois"]

selection = st.radio("", selections,index=4)

if selection == selections[0]: 
    st.write(data1)
elif  selection == selections[1]:
    st.write(data2)
elif  selection == selections[2]:
    st.write(data3)
elif  selection == selections[3]:
    st.write(data4)
elif  selection == selections[4]:
    st.write(data5)
 
    st.markdown("""
               Nous observons que le Random Forest obtient les meilleurs résultats mais nous pouvons encore les améliorer en optimisant les hyperparamètres.
                     """)
st.markdown(""" 
        - L'optimisation
          """)
 
 #rf_loaded = pickle.load(open('.\models\Random Forest.sav', 'rb'))
 # grid_rf = pickle.load(open('.\models\Grid_Random Forest.sav', 'rb'))   





# @st.cache(suppress_st_warning=True)
# @st.cache(allow_output_mutation=True)

def optimisation_models():
    grid_rf = pickle.load(open('.\models\Grid_Random Forest.sav', 'rb'))
    # Optimisation du modèle
    # rf = RandomForestClassifier(random_state=123) 
    # param_grid_rf = [{ 'n_estimators' : [1000] ,
    # 'min_samples_leaf' :  [1 ] ,
    # 'max_features' :  ['sqrt'  ]}] 
    # param_grid_rf = [{ 'n_estimators' : [ 10 , 50 , 100 , 250 , 500 , 1000 ],
    # 'min_samples_leaf' : [ 1 , 3 , 5 ],
    # 'max_features' : [ 'sqrt' , 'log2' ]}] 
    # grid_rf = GridSearchCV(estimator=rf_loaded, param_grid=param_grid_rf)
    # Entraînement du modèle
    # grid_rf = pickle.load(open('.\models\Grid_Random Forest.sav', 'rb'))
    grid_rf.fit(X_train, y_train)
    # with open('.\models\Grid_Random Forest.sav','wb') as f:
    #     pickle.dump( grid_rf,f)
    st.write("Les meilleurs paramètres sont de : {}".format(grid_rf.best_params_))
    st.write("Le score du Random Forest est de : {}".format(grid_rf.score(X_test, y_test)))
  
    
    # Prédiction du modèle
    #y_pred_rf = pickle.load(open('.\models\y_pred_rf_sauv.csv', 'rb'))
    y_pred_rf = grid_rf.predict(X_test) 
    #with open('.\models\y_pred_rf_sauv.csv','wb') as f:
        #pickle.dump( y_pred_rf,f)
    
    st.text('Rapport de classification:\n ' + classification_report(y_test, y_pred_rf))
    return   y_pred_rf,grid_rf,st.write("Les meilleurs paramètres sont de : {}".format(grid_rf.best_params_)),
    st.write("Le score du Random Forest est de : {}".format(grid_rf.score(X_test, y_test))),st.text('Rapport de classification:\n ' + classification_report(y_test, y_pred_rf))
  
if st.button('Hyperparamètres'):
    optimisation_models() 
    st.markdown("""
              Le score est de 0.88 en optimisant meilleur que les prédictions des bookmakers nous avons donc répondu à notre première problématique "Battre les bookmakers c'est donc bien possible".
              L’accuracy calculée sur le jeu de validation est de 88 % , elle concerne l’ensemble des
              prédictions, qu’il s’agisse de la prédiction des victoires ou des défaites.
           Le F1-score  est aussi proche des 90 % ce qui suggère que notre modèle
              est relativement robuste et équilibré. 
            
              Il nous reste désormais à déterminer la stratégie de paris.  """)       



#title = "Stratégie"
sidebar_name = "Stratégie et Conclusion"

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
#data[['Bkm_prediction','Victoire_reel','Winner']]
data["Bkm_predict_vict"] = data['Bkm_prediction'].replace({"D":0,"V":1}).astype(float)


col1,col2= st.columns([1,1])

with col1:
#st.markdown(
    mise_de_depart = st.number_input('Mise de départ',1,100,10)

    start_date, end_date = st.date_input('Choisir date de début, date de fin :', [datetime(2016,1,1),datetime(2018,3,4)],key=0)

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



  
data["Mise"] =  data['Bkm_predict_vict'] * mise_de_depart
data["Gain"] =  data["Mise"] * (data["B365W"] -1 )

st.write("La somme pariée serait de", round(data["Mise"].sum(axis=0),2), "euros et le gain prédit de", round(data["Gain"].sum(axis=0),2),"euros.")
st.write("Soit",round( round(data["Gain"].sum(axis=0),2)/round(data["Mise"].sum(axis=0)),2)*100,"% de bénéfices.Ce gain représente la somme gagnée si nous suivons les préco du bookmaker B365.")
















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

#greater than the start date and smaller than the end date
mask_var = (data_var['Year'] > start_date_var) & (data_var['Year'] <= end_date_var)
data_var_mask = data_var.loc[mask_var]













def paris1( mise = mise_de_depart):
        
    gain = data_var_mask.merge(y_test, how='inner', left_index=True, right_index=True)
    gain = gain[gain["Win"]==1].sum* mise_de_depart * (gain['B365']-1)
    mise_totale = gain[gain["Win"]==1].sum*mise_de_depart  
        
    return  mise_totale, gain,            



# Option 2 : stratégie avec pari uniquement sur la prédiction du modèle pour les gagnants 



 





grid_rf = pickle.load(open('.\models\Grid_Random Forest.sav', 'rb'))
probs = grid_rf.predict_proba(X_test)

def paris2(gain = 0,mise_totale = 0 , mise_de_depart =  mise_de_depart , seuil = 0.8 ):
    y_pred_proba = probs
    for i,probas in enumerate (y_pred_proba):
        cotes = data_var['B365'].iloc[i]
        if probas[1] >= seuil :
            if y_test.iloc[i]== 1:
                gain += round((mise_de_depart * ( probas[1] - seuil ) / ( 1 - seuil )) * (cotes - 1))
            mise_totale += round(mise_de_depart * ( probas[1] - seuil ) / ( 1 - seuil ))
    st.write("La somme pariée serait de", mise_totale, "euros et le gain prédit de", gain,"euros.")
    st.write("Soit",round( gain/mise_totale,2)*100,"% de bénéfices")



paris2()


 

