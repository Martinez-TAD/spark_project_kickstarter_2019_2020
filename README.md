# Spark project MS Big Data Télécom : Kickstarter campaigns

Le but du projet est de faire l'estimation de la réussite d'un projet kickstarter à partir de données 

## Résultats en sortie de TP
En suivant les instructions TP, nous obtenons les prédictions suivantes:

|final_status|predictions|count|
|------------|-----------|-----|
|           1|        0.0| 1058|
|           0|        1.0| 2770|
|           1|        1.0| 2437|
|           0|        0.0| 4456|



|Mesure|Value       |
|------------|-----------|
|F1 precision|0.654|
|Recall      |0.643|
|Precision   |0.697|

Avec la grille, nous obtenons les résultats suivants:


## Amélioration des résultats
Une analyse des données de "goal" montre une forte hétérogénéité des données avec des valeurs très forte.

count    1.081290e+05
mean     3.672623e+04
std      9.719027e+05
min      1.000000e-02
25%      2.000000e+03
50%      5.000000e+03
75%      1.300000e+04
max      1.000000e+08

Nous allons donc faire un filtre sur les valeurs les plus hautes. Nous limiterons les valeurs au quantille 95 "goal", soit la valeur 70000


