 Utilisation des algorithmes de classifications pour reconnaître l'activité humaine
  

 **1. les activités :**
1 MARCHER
2 MARCHER À L'ÉTAGE
3 MARCHER EN DESCENTE
4 ASSIS
5 DEBOUT
6 POSE(allongé)

 **2. les algorithme de classifications**
   K-NN
   Tree
   SVM
   MLP
   NB
  10-fold cross validation comme méthode de validation
  
 **3. Resultats**
10-NN ALGO 
cv=10 
Precision=0.9103002441782835  
Temps=36.39 s           

--------------------------
Tree ALGO 
cv=10 
Precision=0.8677449277262987  
Temps=40.767 s           

--------------------------
SVM ALGO 
cv=10 
Precision=0.9193966024924233  
Temps=75.563 s           

--------------------------
MLP ALGO 
cv=10 
Precision=0.9386072998352629  
Temps=44.129 s           

--------------------------
Gaussian Naive Bayes ALGO 
cv=10 
Precision=0.6978641751867267 
Temps=1.129 s           

--------------------------

![graphe de comparaison entre les algorithmes selon la précision](https://raw.githubusercontent.com/fr33dz/scikitlearn-project/Human-activity-recognition/algo.png)



