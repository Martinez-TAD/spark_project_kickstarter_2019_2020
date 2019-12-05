# Spark project MS Big Data Télécom : Kickstarter campaigns

Le but du projet est de faire l'estimation de la réussite d'un projet kickstarter à partir de données 

## Résultats en sortie de TP
En suivant les instructions TP, nous obtenons les prédictions suivantes:



+------------+-----------+-----+
|final_status|predictions|count|
+------------+-----------+-----+
|           1|        0.0| 1893|
|           0|        1.0| 2352|
|           1|        1.0| 1563|
|           0|        0.0| 5013|
+------------+-----------+-----+

F1 precision = 0.613612798303662
Recall = 0.6077072359301359
Precision = 0.6215632083352709
And with a grid
+------------+-----------+-----+
|final_status|predictions|count|
+------------+-----------+-----+
|           1|        0.0| 1057|
|           0|        1.0| 2873|
|           1|        1.0| 2399|
|           0|        0.0| 4492|
+------------+-----------+-----+

F1 precision = 0.6490644403666332
Recall = 0.6368172996950374
Precision = 0.6963050534403944

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


