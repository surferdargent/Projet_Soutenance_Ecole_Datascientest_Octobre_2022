import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import io
import pickle
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from tabs.second_tab import data_processor

# utiliser la variable processed_data comme ceci:


title = "Préprocessing et Modélisation"
sidebar_name = "Préprocessing et Modélisation"


def run():

    st.title(title)

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
    new_df_preprocessing = data_processor.processed_data
    # new_df_preprocessing = pd.read_csv('df_variables_enrichies.csv',parse_dates=['Year'])
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
        
        Comme le montre le dataframe ci-dessous les variables sont bien toutes au format numérique nous pouvons définir nos features :chart_with_upwards_trend:, notre target :dart: et normaliser nos variables .

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
    @st.cache_data()
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

    
    
    
    
    # Définition du modèle


    


    # Exécution des modèles
    @st.cache_data()     
    def train_model():
        models = []
        models.append(('Logistic Regression',LogisticRegression(random_state=123)))
        models.append(('KNeighbors', KNeighborsClassifier()))
        models.append(('Random Forest', RandomForestClassifier(random_state=123)))
        accuracies = []
        names = []
        trained_models = {}  # Dictionnaire pour stocker les modèles formés
    
        for name, model in models:
            model.fit(X_train,y_train)
            accuracy = model.score(X_test,y_test)
            accuracies.append(accuracy)
            names.append(name)
            trained_models[name] = model  # Stocker le modèle formé
            msg = "Résultat pour %s: %f" % (name, accuracy)
            st.write(msg)
    
        fig = plt.figure()
        sns.barplot(x=names, y=accuracies)
        plt.show()
        st.pyplot(fig)  
    
        return trained_models  # Retourner le dictionnaire des modèles formés

    trained_models = train_model()
    
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
    
    @st.cache_data()
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
      models = {
        "Logistic Regression": LogisticRegression(random_state=123),
        "KNeighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=123)
        }
    
      accuracies = []
      names = []
      data=[]
      model_dict = {} # Dictionnaire pour stocker les modèles formés

        
      for name, model in models.items():
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            accuracies.append(accuracy)
            names.append(name)
            data.append([names, accuracies]) 
    
            model_dict[name] = model # Stocker le modèle dans le dictionnaire
    
            df = pd.DataFrame(list(zip(names, accuracies)), columns=['Noms', f"Scores {x}"])
            
      return df, model_dict

    
    def importance_variables(model_dict):
         fig1 = plt.figure(figsize=(14,6))
         train_features = X_train
         rf_model = model_dict["Random Forest"]  # Récupérer le modèle du dictionnaire
         vars_imp = pd.Series(rf_model.feature_importances_,index=train_features.columns).sort_values(ascending=False)
         sns.barplot(x=vars_imp.index,y=vars_imp)
         plt.xticks(rotation=90)
         plt.xlabel('Variables')
         plt.ylabel("Scores d'importance de la variable")
            
         plt.show()
         st.write("""Ci dessous l'importance des variables nous donne des indications sur le poids de chaque variable sur notre modèle.
         """)  
         return st.pyplot(fig1)
     
    data = new_df_preprocessing_demo
    option = 1
    # Utilisation
    df, model_dict = split_normalisation(data, option)
  
    importance_variables(model_dict)
     
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
    df1, model_dict1 = split_normalisation(df1,1)

    
    # "Features d'origine + Points ELO"  à conserver
    drop_variable2 = ['RankDiff',
    'Ratio_tours_6_mois','Ratio_victoire_6_mois', 'Ratio_surface_6_mois', 'Ratio_tournois_6_mois' ,
    'Ratio_tournois_18_mois', 'Ratio_tours_18_mois','Ratio_victoire_18_mois', 'Ratio_surface_18_mois']
    df2 = new_df_preprocessing_demo.drop(drop_variable2,axis=1)
    df2, model_dict1 = split_normalisation(df2,2)
    

    #"Features d'origine + Points ELO + Diff. de classement "  à conserver
    drop_variable3 = ['Ratio_tours_6_mois','Ratio_victoire_6_mois', 'Ratio_surface_6_mois', 'Ratio_tournois_6_mois' ,
    'Ratio_tournois_18_mois', 'Ratio_tours_18_mois','Ratio_victoire_18_mois', 'Ratio_surface_18_mois']
    df3 = new_df_preprocessing_demo.drop(drop_variable3,axis=1)
    df3, model_dict1 = split_normalisation(df3,3)


    #"Features d'origine + Points ELO + Diff. de classement + Moy.roulantes 6 mois"  à conserver
    drop_variable4 = ['Ratio_tournois_18_mois', 'Ratio_tours_18_mois','Ratio_victoire_18_mois', 'Ratio_surface_18_mois']
    df4 = new_df_preprocessing_demo.drop(drop_variable4,axis=1)
    df4, model_dict1 = split_normalisation(df4,4)


    #"Features d'origine + Points ELO + Diff. de classement + Moy.roulantes 6 mois + Moy.roulantes 18 mois"  à conserver
    df5 = new_df_preprocessing_demo  
    df5, model_dict1 = split_normalisation(df5,5)

    






    
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
    
    
    @st.cache_data()
    def creat_df():
         
         data1 =  pd.concat([df6,df1],axis=1)
          
         data2 =  pd.concat([df6,df1,df2],axis=1)
          
         data3 =  pd.concat([df6,df1,df2,df3],axis=1)
          
         data4 =  pd.concat([df6,df1,df2,df3,df4],axis=1)
          
         data5 =  pd.concat([df6,df1,df2,df3,df4,df5],axis=1)
         return (data1,data2,data3,data4,data5)
    data1,data2,data3,data4,data5 = creat_df()
            
       
    data1 =  pd.concat([df6,df1],axis=1)

    data2 =  pd.concat([df6,df1,df2],axis=1)

    data3 =  pd.concat([df6,df1,df2,df3],axis=1)

    data4 =  pd.concat([df6,df1,df2,df3,df4],axis=1)

    data5 =  pd.concat([df6,df1,df2,df3,df4,df5],axis=1)
    selections = ["Features d'origine" , "Features d'origine + Pts ELO", "Features d'origine + Pts ELO + Diff. de classement","Features d'origine + Pts ELO + Diff. de classement + Moy.roulantes 6 mois","Features d'origine + Pts ELO + Diff. de classement + Moy.roulantes 6 mois + Moy.roulantes 18 mois"]

    selection = st.radio("", selections,index= 4)

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
    
  


    @st.cache_data()     
    def optimisation_models():
            param_grid = {
                'n_estimators': [1000],
                'max_features': ['sqrt'],
                'min_samples_leaf' :  [1 ] 
            }
            grid_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=123), param_grid=param_grid, cv= 5)
            grid_rf.fit(X_train, y_train)
            trained_models['Grid_Random Forest'] = grid_rf  # Stocker le modèle formé dans le dictionnaire
            st.write("Les meilleurs paramètres sont de : {}".format(grid_rf.best_params_))
            st.write("Le score du Random Forest est de : {}".format(grid_rf.score(X_test, y_test)))
            y_pred_rf = grid_rf.predict(X_test) 
            st.text('Rapport de classification:\n ' + classification_report(y_test, y_pred_rf))
            return y_pred_rf, grid_rf


    if st.button('Hyperparamètres'):
            optimisation_models() 
            st.markdown("""
                     Le score est de 0.88 en optimisant meilleur que les prédictions des bookmakers nous avons donc répondu à notre première problématique "Battre les bookmakers c'est donc bien possible".
                     L’accuracy calculée sur le jeu de validation est de 88 % , elle concerne l’ensemble des
                     prédictions, qu’il s’agisse de la prédiction des victoires ou des défaites.
                     Le rapport entre la précision et le rappel est d’environ 1, ce qui suggère que notre modèle
                     est relativement robuste et équilibré. 
                   
                     Il nous reste désormais à déterminer la stratégie de paris.  """)       
    
    







 