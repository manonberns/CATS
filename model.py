import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split

#Variables
TEST_SIZE = 0.1
NUMBER_OF_SAMPLES_CV = 90
NUMBER_OF_FEATURES_CV = 100
NUMBER_OF_CLASSES_CV = 3

# import data
def import_data():
    data=pd.read_table('Complied-Data.txt', sep='\t', delimiter=None, delim_whitespace=False, header=0, index_col=0)
    X = (data.iloc[0:100, 1:-1])
    y = data.iloc[0:100,0]
    return(X,y)

# returns train and test set
def split_data():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=0)
    return(X_train, X_test, y_train, y_test)
    
def SVC_crossvalidation(X,y):
    # Build a classification task using 3 informative features
    X, y = make_classification(n_samples=NUMBER_OF_SAMPLES_CV, n_features=NUMBER_OF_FEATURES_CV, n_informative=3,
                               n_redundant=2, n_repeated=0, n_classes=NUMBER_OF_CLASSES_CV,
                               n_clusters_per_class=1, random_state=0)
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring='accuracy')
    rfecv.fit(X, y)
    print("Optimal number of features : %d" % rfecv.n_features_)
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

# main
X,y = import_data()
X_train, X_test, y_train, y_test = split_data()
SVC_crossvalidation(X_train, y_train)

