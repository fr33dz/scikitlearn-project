import time, os, sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score


path = os.getcwd()
# Afectation de X (Variables) et y (Sorties)
X = np.loadtxt(path+'/data/X_train.txt')
y = np.loadtxt(path+'/data/y_train.txt')

##Affichage 
def report(algo, cv, pecision, temps):
    print("{} ALGO \ncv={} \nPrecision={}  \nTemps={} s \
          ".format(algo, cv , pecision , round(temps, 3)))
    print("--------------------------")




K=10
CV = 10


start_time = time.time()
#Application de l'algo KNN avec K=4 et la validation croisée 10 iterations
knn = KNeighborsClassifier(n_neighbors=K)
scores = cross_val_score(knn, X, y, cv=CV, scoring='accuracy')
plt.plot(scores, label="10-NN")
#Affichage 
report("10-NN",CV ,scores.mean(), (time.time() - start_time))
#sys.exit(0)

from sklearn import tree
start_time = time.time()
clf_tree = tree.DecisionTreeClassifier()
scores = cross_val_score(clf_tree, X, y, cv=CV, scoring='accuracy')
plt.plot(scores, label="TREE")
#Affichage 
report("Tree",CV ,scores.mean(), (time.time() - start_time))

from sklearn import svm
start_time = time.time()
clf_svm = svm.SVC()
scores = cross_val_score(clf_svm, X, y, cv=CV, scoring='accuracy')
plt.plot(scores, label="SVM")
##Affichage 
report("SVM",CV ,scores.mean(), (time.time() - start_time))

from sklearn.neural_network import MLPClassifier
start_time = time.time()
clf_MLP = MLPClassifier()
scores = cross_val_score(clf_MLP, X, y, cv=CV, scoring='accuracy')
plt.plot(scores, label="MLP")
##Affichage 
report("MLP",CV ,scores.mean(), (time.time() - start_time))

from sklearn.naive_bayes import GaussianNB
start_time = time.time()
clf_gnb = GaussianNB()
scores = cross_val_score(clf_gnb, X, y, cv=CV, scoring='accuracy')
plt.plot(scores, label="GHB")
##Affichage 
report("Gaussian Naive Bayes",CV ,scores.mean(), (time.time() - start_time))

plt.ylabel('Precision')
plt.xlabel('Iterations')
plt.title('Comparaison des precisions')
plt.legend()
plt.show()
