import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import time

# Afectation de X (Variables) et y (Sorties)
X = np.loadtxt('E:/TP MATLAB/tp2/tp2 python/X_train.txt')
y = np.loadtxt('E:/TP MATLAB/tp2/tp2 python/y_train.txt')

##Affichage 
def report(algo, cv, pecision, temps):
    print("{} ALGO \ncv={} \nPrecision={}  \nTemps={} s \
          ".format(algo, cv , round(pecision, 3), round(temps, 3)))
    print("--------------------------")

K=10
CV = 10
tab_scores = {}

start_time = time.time()
#Application de l'algo KNN avec K=4 et la validation croisée 10 iterations
knn = KNeighborsClassifier(n_neighbors=K)
scores = cross_val_score(knn, X, y, cv=CV, scoring='recall')
plt.plot(scores, label="10-NN")
#Affichage 
report("10-NN",CV ,scores.mean(), (time.time() - start_time))

from sklearn import tree
start_time = time.time()
clf_tree = tree.DecisionTreeClassifier()
scores = cross_val_score(clf_tree, X, y, cv=CV, scoring='recall')
plt.plot(scores, label="TREE")
#Affichage 
report("Tree",CV ,scores.mean(), (time.time() - start_time))

from sklearn import svm
start_time = time.time()
clf_svm = svm.SVC()
scores = cross_val_score(clf_svm, X, y, cv=CV, scoring='recall')
plt.plot(scores, label="SVM")
##Affichage 
report("SVM",CV ,scores.mean(), (time.time() - start_time))

from sklearn.neural_network import MLPClassifier
start_time = time.time()
clf_MLP = MLPClassifier()
scores = cross_val_score(clf_MLP, X, y, cv=CV, scoring='recall')
plt.plot(scores, label="MLP")
##Affichage 
report("MLP",CV ,scores.mean(), (time.time() - start_time))

from sklearn.naive_bayes import GaussianNB
start_time = time.time()
clf_gnb = GaussianNB()
scores = cross_val_score(clf_gnb, X, y, cv=CV, scoring='recall')
plt.plot(scores, label="GHB")
##Affichage 
report("Gaussian Naive Bayes",CV ,scores.mean(), (time.time() - start_time))

plt.ylabel('Precision')
plt.xlabel('Iterations')
plt.title('Comparaison des precisions')
plt.legend()
plt.show()
