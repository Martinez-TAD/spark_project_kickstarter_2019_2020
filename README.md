# Spark project MS Big Data Télécom : Kickstarter campaigns

Le but du projet est de faire l'estimation de la réussite d'un projet kickstarter à partir de données 

## Résultats en sortie de TP
En suivant les instructions TP, nous obtenons les prédictions suivantes:

|final_status|predictions|count|
|------------|-----------|-----|
|	    1|        0.0| 1952|
|           0|        1.0| 2345|
|           1|        1.0| 1504|
|           0|        0.0| 5020|

|Mesure        | Value|
|--------------|------|
|F1 precision| **0.6081**|
|Recall         |0.6029|
|Precision     |0.6148|

*Avec la grille, nous obtenons les valeurs suivantes:*

|final_status|predictions|count|
|------------|-----------|-----|
|           1|        0.0| 1060|
|           0|        1.0| 2857|
|           1|        1.0| 2396|
|           0|        0.0| 4508|

|Mesure        | Value|
|--------------|------|
|F1 precision| **0.6502**|
|Recall         |0.6380|
|Precision     |0.6967|

Les meilleurs paramètres sont les suivants: minDF=55, regParam: 1.0E-8


## Amélioration des résultats

### Filtrage des outliners
Une analyse des données de "goal" montre une forte hétérogénéité des données avec des valeurs très forte.

count    1.081290e+05
mean     3.672623e+04
std      9.719027e+05
min      1.000000e-02
25%      2.000000e+03
50%      5.000000e+03
75%      1.300000e+04
max      1.000000e+08

Nous allons donc faire un filtre sur les valeurs les plus hautes. Nous limiterons les valeurs au quantile 95% , soit la valeur 70000


|Mesure        | Sans Grille| Avec une Grille|
|--------------|------|------|
|F1 precision| **0.6172**|**0.6458**
|Recall         |0.6117|0.6347
|Precision     |0.6248|0.6988

Le filtrage des valeurs outliners de "goal" ne semble pas améliorer la qualité du modèle.

### Extension de la grille
J'ai fais une extension de la grille pour voir si des paramètres différents peuvent influer sur le résultat. J'ai notament ajouté le paramètres de l'elasticnet dans l'équation.
J'obtiens les résultats suivants:

|Mesure        |  Grille initiale| Grille étendue|
|--------------|-----------------|-------------|
|F1 precision| **0.6502**| **0.6597**| 
|Recall         |0.6380|0.6480|
|Precision     |0.6967|0.7007|

Les meilleures hyper-paramètres sont les suivants:
  elasticNetParam: 0.4,
  minDF: 30.0,
  regParam: 1.0E-8


### Utilisation d'une RandomForest
Nous allons voir pour utiliser une randomForest afin d'etre en non linéaire.

|final_status|predictions|count|
|------------|-----------|-----|
|           1|        0.0| 3456|
|           0|        0.0| 7365|

Les mesures sont claires: le prédicteur prédit toujours la même chose! 
Je monte une grille pour voir si en modifiant des paramètres nous pouvons avoir une vraie prédiction.

*** WIP: la grille tourne depuis 2 jours...*** 
