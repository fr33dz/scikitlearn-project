
import time, os, sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import VotingClassifier
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

##Affichage des mesures 
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
    
    #plt.plot(precision, label=algo) 
    print("--------------------------")


CV = 10 

from sklearn.neighbors import KNeighborsClassifier
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
#Affichage 
report("SVM",CV ,scorer, (time.time() - start_time))


start_time = time.time()
#Application de l'algo VotingClassifier et la validation croisée 10 iterations
clf_vc = VotingClassifier(estimators=[
    ('knn', knn), ('tree', clf_tree), ('svm', clf_svm)],voting='soft')
cross_val_score(clf_vc, X, y, scoring=scorer, cv=10)
#Affichage 
report("VotingClassifier",CV ,scorer, (time.time() - start_time))


plt.plot(precisions[0][0:9], label='KNN')
plt.plot(precisions[0][10:19], label='TREE')

max_precision = max(precisions[0][30:39])

plt.ylabel('Précision')
plt.xlabel('Itérations')
plt.title('Comparaison des precisions')
plt.legend()
plt.show()
