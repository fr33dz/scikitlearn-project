import time, os, sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
from multiscorer import MultiScorer


path = os.getcwd()
# Affectation de X (Variables) et y (Sorties)
X = np.loadtxt(path+'/data/X_train.txt')
y = np.loadtxt(path+'/data/y_train.txt')


scorer = MultiScorer({
'accuracy' : (accuracy_score,  {}),
'precision': (precision_score,{'average': 'macro'}),
'recall'   : (recall_score,   {'average': 'macro'}),
'AUC'      : (auc,             {'reorder': True}),
'F-measure': (f1_score,        {'average': 'macro'})
})



precisions = []

##Affichage 
def report(algo, cv, scorer, temps):
    print("ALGO : {}  \ncv : {}  \nTemps : {} s \
          ".format(algo, cv , round(temps, 3)))
    results = scorer.get_results()
    for metric_name in results.keys():
        average_metric_score = np.average(results[metric_name])
        if metric_name == 'AUC':
            print('Average AUC score : %d' % (average_metric_score))
        else:
            print('Average %s score :  %.3f' % (metric_name, average_metric_score))

    precision = results['precision']
    #plt.plot(precision, label=algo)
    precisions.append(precision)
    precision = 0
    print("--------------------------")


CV = 10 

start_time = time.time()
#Application de l'algo KNN avec K=4 et la validation croisée 10 iterations
knn = KNeighborsClassifier(n_neighbors=10)
cross_val_score(knn, X, y, scoring=scorer, cv=10)
#Affichage 
report("10-NN",CV ,scorer, (time.time() - start_time))
#sys.exit(1)

from sklearn import tree
start_time = time.time()
clf_tree = tree.DecisionTreeClassifier()
scores = cross_val_score(clf_tree, X, y,scoring=scorer, cv=10)
#Affichage 
report("Tree",CV ,scorer, (time.time() - start_time))

from sklearn import svm
start_time = time.time()
clf_svm = svm.SVC()
scores = cross_val_score(clf_svm, X, y, scoring=scorer, cv=10)
##Affichage 
report("SVM",CV ,scorer, (time.time() - start_time))

from sklearn.neural_network import MLPClassifier
start_time = time.time()
clf_MLP = MLPClassifier()
scores = cross_val_score(clf_MLP, X, y,scoring=scorer, cv=10)
##Affichage 
report("MLP",CV ,scorer, (time.time() - start_time))

from sklearn.naive_bayes import GaussianNB
start_time = time.time()
clf_gnb = GaussianNB()
scores = cross_val_score(clf_gnb, X, y, scoring=scorer, cv=10)
##Affichage 
report("Gaussian Naive Bayes",CV ,scorer, (time.time() - start_time))

plt.ylabel('Précision')
plt.xlabel('Itérations')
plt.title('Comparaison des precisions')
plt.legend()
plt.show()
