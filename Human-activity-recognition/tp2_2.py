import time, os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
from multiscorer import MultiScorer


path = os.getcwd()
#fectation de X (Covariables) et y (Sorties)
X = np.loadtxt(path+'/data/X_train.txt')
y = np.loadtxt(path+'/data/y_train.txt')

#Evaluation et mesures de perfermances 
scorer = MultiScorer({
'accuracy' : (accuracy_score,  {}),
'precision': (precision_score,{'average': 'macro'}),
'recall'   : (recall_score,   {'average': 'macro'}),
'AUC'      : (auc,             {'reorder': True}),
'F-measure': (f1_score,        {'average': 'macro'})
})


#Tableau de precesion de chaque Algo 
precisions = []

#Affichage des mesures 
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
        if metric_name == 'precision':
            precision = results['precision']
            precisions.append(precision)
            precision = 0
    print("--------------------------")


CV = 10
#Application de l'algo KNN avec K=4 et la validation croisée 10 iterations
from sklearn.neighbors import KNeighborsClassifier
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=4)
cross_val_score(knn, X, y, scoring=scorer, cv=10)
#Affichage 
report("10-NN",CV ,scorer, (time.time() - start_time))


#Application de l'algo tree et la validation croisée 10 iterations
from sklearn.tree import DecisionTreeClassifier
start_time = time.time()
clf_tree = DecisionTreeClassifier()
scores = cross_val_score(clf_tree, X, y,scoring=scorer, cv=10)
#Affichage 
report("Tree",CV ,scorer, (time.time() - start_time))

#Application de l'algo SVM et la validation croisée 10 iterations
from sklearn import svm
start_time = time.time()
clf_svm = svm.SVC()
scores = cross_val_score(clf_svm, X, y, scoring=scorer, cv=10)
#Affichage 
report("SVM",CV ,scorer, (time.time() - start_time))


#Application de l'algo MPL et la validation croisée 10 iterations
from sklearn.neural_network import MLPClassifier
start_time = time.time()
clf_MLP = MLPClassifier()
scores = cross_val_score(clf_MLP, X, y,scoring=scorer, cv=10)
#Affichage 
report("MLP",CV ,scorer, (time.time() - start_time))

#Application de l'algo GNB et la validation croisée 10 iterations
from sklearn.naive_bayes import GaussianNB
start_time = time.time()
clf_gnb = GaussianNB()
scores = cross_val_score(clf_gnb, X, y, scoring=scorer, cv=10)
#Affichage 
report("Gaussian Naive Bayes",CV ,scorer, (time.time() - start_time))

#Dessiner le graphe 
plt.plot(precisions[0][0:9], label='KNN')
plt.plot(precisions[0][10:19], label='TREE')
plt.plot(precisions[0][20:29], label='SVM')
plt.plot(precisions[0][30:39], label='MPL')
plt.plot(precisions[0][40:49], label='GNB')

#Annoter la meilleure precision sur le graphe
meilleur_precision = max(precisions[0][30:39])
num_iteration = (precisions[0][30:39]).index(meilleur_precision)
plt.annotate('['+str(num_iteration)+', '+ str(round(meilleur_precision, 3))+']', xy=(num_iteration, meilleur_precision), xytext=(3.5, 0.98),
            arrowprops=dict(facecolor='purple', shrink=0.01),)
plt.ylabel('Précision')
plt.xlabel('Itérations')
plt.title('Comparaison des precisions')
plt.legend()
plt.show()
