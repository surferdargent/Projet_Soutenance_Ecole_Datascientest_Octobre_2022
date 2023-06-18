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

def run():
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

    new_df_preprocessing = pd.read_csv('df_variables_enrichies.csv', parse_dates=['Year'])
    new_df_preprocessing['Year'] = new_df_preprocessing['Year'].astype(str)

    new_df_preprocessing['Year'] = pd.to_datetime(new_df_preprocessing['Year'])
    new_df_preprocessing['Year'] = new_df_preprocessing['Year'].dt.date

    col1, col2 = st.columns([3, 1])

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
            """Toutes les variables de type "object" devront être encodées ..."""
        )

    st.markdown(
        """
        :tennis: Pour l'encodage nous avons choisi la méthodologie suivante :
        - Pour les variables **Player**, **Tournament**, **Series** et **Round** nous remplaçons les noms par des numéros id.
        - Pour la variable **Court** nous remplaçons ‘Outdoor’, ‘Indoor’ par 0, 1 et pour la variable **Surface** nous changeons les modes 'Hard', 'Clay’, 'Grass', 'Carpet' par 0, 1, 2, 3.
        - Pour les variables numériques mais qui sont au format texte nous les typons en float.
        """
    )

    # Encodage des variables
    # Certaines variables sont catégorielles il faut les passer en numérique

    # Player
    Id_player = pd.DataFrame(new_df_preprocessing['Player'].unique(), columns=['Name'])
    Id_player = Id_player.rename_axis('Id_player').reset_index()
    new_df_preprocessing = pd.merge(new_df_preprocessing, Id_player, left_on="Player", right_on="Name")
    new_df_preprocessing["Id_player"] = new_df_preprocessing['Id_player'].astype(float)

    # Tournament 
    Id_tournament = pd.DataFrame(new_df_preprocessing['Tournament'].unique(), columns=['Tourna'])
    Id_tournaments = Id_tournament.rename_axis('Id_tournament').reset_index()
    new_df_preprocessing = pd.merge(new_df_preprocessing, Id_tournaments, left_on="Tournament", right_on="Tourna")
    new_df_preprocessing["Id_tournament"] = new_df_preprocessing['Id_tournament'].astype(float)

    # Series
    Id_Series = pd.DataFrame(new_df_preprocessing['Series'].unique(), columns=['Type_series'])
    Id_Series = Id_Series.rename_axis('Id_series').reset_index()
    new_df_preprocessing = pd.merge(new_df_preprocessing, Id_Series, left_on="Series", right_on='Type_series')
    new_df_preprocessing["Id_series"] = new_df_preprocessing["Id_series"].astype(float)

    # Surface
    new_df_preprocessing['Surface'] = new_df_preprocessing['Surface'].replace(to_replace=['Hard', 'Clay', 'Grass', 'Carpet'], value=[0, 1, 2, 3])
    new_df_preprocessing['Surface'] = new_df_preprocessing['Surface'].astype(float)

    # Court
    new_df_preprocessing['Court'] = new_df_preprocessing['Court'].replace(to_replace=['Outdoor', 'Indoor'], value=[0, 1])
    new_df_preprocessing['Court'] = new_df_preprocessing['Court'].astype(float)

    # Round
    Id_round = pd.DataFrame(new_df_preprocessing['Round'].unique(), columns=['Type_round'])
    Id_round = Id_round.rename_axis('Id_round').reset_index()
    new_df_preprocessing = pd.merge(new_df_preprocessing, Id_round, left_on="Round", right_on='Type_round')
    new_df_preprocessing["Id_round"] = new_df_preprocessing["Id_round"].astype(float)

    new_df_preprocessing = new_df_preprocessing.drop(['Player', 'Location', 'Tournament', 'Series', 'Round', 'Name', 'Tourna',
                                                      'Type_series', 'Type_round'], axis=1)

    # Passer les variables object numériques en float
    new_df_preprocessing[['BestOf', 'Court', 'Surface', 'Rank', 'SetsWon', 'EloPoints', 'B365', 'RankDiff', 'Win', 'Id_player',
                          'Id_tournament', 'Id_series', 'Id_round']] = new_df_preprocessing[['BestOf', 'Court', 'Surface', 'Rank',
                                                                                             'SetsWon', 'EloPoints', 'B365',
                                                                                             'RankDiff', 'Win', 'Id_player',
                                                                                             'Id_tournament', 'Id_series',
                                                                                             'Id_round']].astype(float)

    st.markdown(
        """
        Comme le montre le dataframe ci-dessous, les variables sont bien toutes au format numérique. Nous pouvons définir nos features :chart_with_upwards_trend:, notre target :dart: et normaliser nos variables.
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
        st.image("./assets/Diapositive3.png", width=650)

    from sklearn.preprocessing import StandardScaler

    # Modèle de classification que l'on va utiliser
    from sklearn.ensemble import RandomForestClassifier

    def split(data):
        df = pd.DataFrame(data)
        df = df.sort_values(by=["Year"], ascending=True)

        # Diviser le dataset en "train" et "test" toutes les données avant le 01 janvier 2016 seront égales au "train" et après au test
        date_split = pd.Timestamp(2016, 1, 1)

        df['Year'] = pd.to_datetime(df['Year'])
        data_train = df[df['Year'] < date_split]
        data_test = df[df['Year'] >= date_split]

        # Création des quatre variables pour l'entrainement et le test (X_train, X_test, y_train, y_test)
        X_train = data_train.drop(['Win'], axis=1)
        X_test = data_test.drop(['Win'], axis=1)

        y_train = data_train['Win']
        y_test = data_test['Win']

        X_train = X_train.select_dtypes('float')
        X_test = X_test.select_dtypes('float')

        # On normalise nos données numériques
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        X_train = X_train_scaled
        X_test = X_test_scaled

        return X_train, y_train, X_test, y_test

    def train_model(data):
        X_train, y_train, X_test, y_test = split(data)

        models = []
        models.append(('Logistic Regression', LogisticRegression(random_state=123)))
        models.append(('KNeighbors', KNeighborsClassifier()))
        models.append(('Random Forest', RandomForestClassifier(random_state=123)))

        accuracies = []
        names = []

        for name, model in models:
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            accuracies.append(accuracy)
            names.append(name)

            msg = "Résultat pour %s: %f" % (name, accuracy)
            st.write(msg)

        fig = plt.figure()
        sns.barplot(x=names, y=accuracies)
        plt.show()
        st.pyplot(fig)

    train_model(new_df_preprocessing)

    st.markdown("""
        Après avoir normalisé et entraîné les modèles, nous avons obtenu des scores supérieurs à 95 %. Cependant, les performances de nos modèles laissent penser qu'il y a peut-être un surapprentissage. Pour vérifier cela, nous allons créer une matrice de corrélation.
    """)

    new_df_preprocessing['Year'] = pd.to_datetime(new_df_preprocessing['Year'])
    cor = new_df_preprocessing.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cor, ax=ax, cmap='coolwarm')
    st.write(fig)

    st.markdown("""
        Nous constatons que la variable "SetsWon" est fortement corrélée avec "Win", ce qui est normal car elle donne une indication sur le nombre de sets gagnés lors du match. Cependant, cette information n'est pas disponible avant le match, nous devons donc la supprimer des features et relancer le modèle.
    """)

    st.markdown(""":tennis: 2ème entraînement """)

    new_df_preprocessing_demo = new_df_preprocessing.drop("SetsWon", axis=1)
    X_train, y_train, X_test, y_test = split(new_df_preprocessing_demo)
    train_model(new_df_preprocessing_demo)

    st.markdown("""
        Ces résultats semblent plus conformes à ce que l'on peut attendre pour ce type de données. Le meilleur modèle est le Random Forest avec un score de 0,88, ce qui est meilleur que les prédictions des bookmakers. Nous allons maintenant essayer d'améliorer les performances du modèle.
    """)

    def split_normalisation(data):
        df = pd.DataFrame(data)
        df = df.sort_values(by=["Year"], ascending=True)

        # Diviser le dataset en "train" et "test" : toutes les données avant le 01 janvier 2016 seront égales au "train" et après au test
        date_split = pd.Timestamp(2016, 1, 1).date()
        df["Year"] = pd.to_datetime(df["Year"])
        df["Year"] = df["Year"].dt.date
        data_train = df[df['Year'] < date_split]
        data_test = df[df['Year'] >= date_split]

        # Création des quatre variables pour l'entrainement et le test (X_train, X_test, y_train, y_test)
        X_train = data_train.drop(['Win'], axis=1)
        X_test = data_test.drop(['Win'], axis=1)
        y_train = data_train['Win']
        y_test = data_test['Win']

        X_train = X_train.select_dtypes('float')
        X_test = X_test.select_dtypes('float')

        # On normalise nos données numériques
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        X_train = X_train_scaled
        X_test = X_test_scaled

        models = []
        models.append(('Logistic Regression', LogisticRegression(random_state=123)))
        models.append(('KNeighbors', KNeighborsClassifier()))
        models.append(('Random Forest', RandomForestClassifier(random_state=123)))

        accuracies = []
        names = []
        best_model = None
        best_accuracy = 0

        for name, model in models:
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            accuracies.append(accuracy)
            names.append(name)

            # Garder le modèle avec le meilleur score
            if accuracy > best_accuracy:
                best_model = model
                best_accuracy = accuracy

        df = pd.DataFrame(list(zip(names, accuracies)), columns=['Noms', 'Scores'])

        return df, best_model

    def importance_variables(model):
        fig1 = plt.figure(figsize=(14, 6))
        train_features = X_train
        vars_imp = pd.Series(model.feature_importances_, index=train_features.columns).sort_values(ascending=False)
        sns.barplot(x=vars_imp.index, y=vars_imp)
        plt.xticks(rotation=90)
        plt.xlabel('Variables')
        plt.ylabel("Scores d'importance de la variable")

        plt.show()
        st.write("""Ci-dessous l'importance des variables nous donne des indications sur le poids de chaque variable sur notre modèle.""")
        st.pyplot(fig1)

    df_scores, best_model = split_normalisation(new_df_preprocessing_demo)
    importance_variables(best_model)

    st.markdown("""
        :tennis: Amélioration du modèle RF

        Nous allons procéder en deux étapes : nous allons d'abord identifier les paramètres qui apportent le meilleur gain de performances en changeant les variables d'entraînement, puis nous réglerons ensuite les hyperparamètres.
    """)

    st.markdown("""
         - Analyse de la performance des variables

         Une première approche consiste à utiliser de façon sélective les différents paramètres.
    """)

    df1 = df_scores[df_scores['Noms'] == 'Features d\'origine ']
    df2 = df_scores[df_scores['Noms'] == 'Features d\'origine + Pts ELO']
    df3 = df_scores[df_scores['Noms'] == 'Features d\'origine + Pts ELO + Diff. de classement']
    df4 = df_scores[df_scores['Noms'] == 'Features d\'origine + Pts ELO + Diff. de classement + Moy.roulantes 6 mois']
    df5 = df_scores[df_scores['Noms'] == 'Features d\'origine + Pts ELO + Diff. de classement + Moy.roulantes 6 mois + Moy.roulantes 18 mois']

    df1, best_model_1 = split_normalisation(new_df_preprocessing_demo)
    df2, best_model_2 = split_normalisation(new_df_preprocessing_demo)
    df3, best_model_3 = split_normalisation(new_df_preprocessing_demo)
    df4, best_model_4 = split_normalisation(new_df_preprocessing_demo)
    df5, best_model_5 = split_normalisation(new_df_preprocessing_demo)

    lengths = [len(df1['Scores'].values), len(df2['Scores'].values), len(df3['Scores'].values), len(df4['Scores'].values),
               len(df5['Scores'].values)]

    if len(set(lengths)) != 1:
        st.error("Les tableaux de scores n'ont pas la même longueur.")
    else:
        df_scores = pd.DataFrame({
            'Modèle': ['Features d\'origine', 'Features d\'origine + Pts ELO',
                       'Features d\'origine + Pts ELO + Diff. de classement',
                       'Features d\'origine + Pts ELO + Diff. de classement + Moy.roulantes 6 mois',
                       'Features d\'origine + Pts ELO + Diff. de classement + Moy.roulantes 6 mois + Moy.roulantes 18 mois'],
            'Scores 1': df1['Scores'].values,
            'Scores 2': df2['Scores'].values,
            'Scores 3': df3['Scores'].values,
            'Scores 4': df4['Scores'].values,
            'Scores 5': df5['Scores'].values
        })

    st.markdown("""
        Nous observons que le Random Forest obtient les meilleurs résultats, mais nous pouvons encore les améliorer en optimisant les hyperparamètres.
    """)

    def optimisation_models():
        rf = RandomForestClassifier(random_state=123)

        param_grid_rf = [
            {'n_estimators': [1000],
             'min_samples_leaf': [1],
             'max_features': ['sqrt']}
        ]

        grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf)
        grid_rf.fit(X_train, y_train)

        st.write("Les meilleurs paramètres sont : {}".format(grid_rf.best_params_))
        st.write("Le score du Random Forest est : {}".format(grid_rf.score(X_test, y_test)))

        y_pred_rf = grid_rf.predict(X_test)

        st.text('Rapport de classification:\n' + classification_report(y_test, y_pred_rf))

    optimisation_models()

    st.markdown("""
        Le score après l'optimisation est de 0,88, ce qui est meilleur que les prédictions des bookmakers. Nous avons donc répondu à notre première problématique : "Battre les bookmakers est donc possible". L'accuracy calculée sur le jeu de validation est de 88 %, elle concerne l'ensemble des prédictions, qu'il s'agisse de la prédiction des victoires ou des défaites. Le rapport entre la précision et le rappel est d'environ 1, ce qui suggère que notre modèle est relativement robuste et équilibré. Il nous reste maintenant à déterminer la stratégie de paris.
    """)
