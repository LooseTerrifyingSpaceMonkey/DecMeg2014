"""DecMeg2014 example code.

Simple prediction of the class labels of the test set by:
- pooling all the triaining trials of all subjects in one dataset.
- Extracting the MEG data in the first 500ms from when the
  stimulus starts.
- Using a linear classifier (logistic regression).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from scipy.io import loadmat
from sklearn.grid_search import GridSearchCV
from time import time


def create_features(XX, tmin, tmax, sfreq, tmin_original=-0.5):
    """Creation of the feature space:
    - restricting the time window of MEG data to [tmin, tmax]sec.
    - Concatenating the 306 timeseries of each trial in one long
      vector.
    - Normalizing each feature independently (z-scoring).
    """
    print "Applying the desired time window."
    beginning = np.round((tmin - tmin_original) * sfreq).astype(np.int)
    end = np.round((tmax - tmin_original) * sfreq).astype(np.int)
    XX = XX[:, :, beginning:end].copy()

    print "2D Reshaping: concatenating all 306 timeseries."
    XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])

    print "Features Normalization."
    XX -= XX.mean(0)
    XX = np.nan_to_num(XX / XX.std(0))

    return XX


if __name__ == '__main__':

    print "DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain"
    print
    subjects_train = range(1, 17) # use range(1, 17) for all subjects
    print "Training on subjects", subjects_train

    # We throw away all the MEG data outside the first 0.5sec from when
    # the visual stimulus start:
    tmin = 0.0
    tmax = 0.500
    print "Restricting MEG data to the interval [%s, %s]sec." % (tmin, tmax)

    X_train = []
    y_train = []
    X_test = []
    ids_test = []

    classifiers = []

    print
    print "Creating the trainset. And training the classifiers for each subject. "
    for subject in subjects_train:
        filename = '../data/mat/train_subject%02d.mat' % subject
        print "Loading", filename
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        yy = data['y']
        sfreq = data['sfreq']
        tmin_original = data['tmin']
        print "Dataset summary:"
        print "XX:", XX.shape
        print "yy:", yy.shape
        print "sfreq:", sfreq

        XX = create_features(XX, tmin, tmax, sfreq)

        X_train.append(XX)
        y_train.append(yy)

        xx = np.vstack(XX)

        '''print
        adaClf = AdaBoostClassifier(n_estimators=100)
        print "Classifier:"
        print adaClf
        print "Training", subject
        adaClf.fit(xx, yy)
        classifiers.append(adaClf)

        print
        rfClf = RandomForestClassifier(n_estimators=1000, max_depth=None, max_features=375, n_jobs=-1)
        print "Classifier:"
        print rfClf
        print "Training", subject
        rfClf.fit(xx, yy)
        classifiers.append(rfClf)

        print
        sgdClf = SGDClassifier(loss="hinge", penalty="l2")
        print "Classifier:"
        print sgdClf
        print "Training", subject
        sgdClf.fit(xx, yy)
        classifiers.append(sgdClf)

        print
        nnClf = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', weights='distance')
        print "Classifier:"
        print nnClf
        print "Training", subject
        nnClf.fit(xx, yy)
        classifiers.append(nnClf)'''

        print
        t0 = time()
        param_grid = {'C': [1, 10, 1e2, 1e3, 1e4, 1e5], 'penalty': ['l1', 'l2']}
        lrClf = GridSearchCV(LogisticRegression(random_state=0), param_grid)
        print "Classifier:"
        print lrClf
        print "Training", subject
        lrClf.fit(xx, yy)
        print "Best estimator found by grid search:"
        print lrClf.best_estimator_
        print "Done in:", (time() - t0)
        classifiers.append(lrClf)

    # X_train is the full stack of subjects
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    print "Trainset:", X_train.shape

    print
    print "Creating the testset."
    subjects_test = range(17, 24)
    for subject in subjects_test:
        filename = '../data/mat/test_subject%02d.mat' % subject
        print "Loading", filename
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        ids = data['Id']
        sfreq = data['sfreq']
        tmin_original = data['tmin']
        print "Dataset summary:"
        print "XX:", XX.shape
        print "ids:", ids.shape
        print "sfreq:", sfreq

        XX = create_features(XX, tmin, tmax, sfreq)

        X_test.append(XX)
        ids_test.append(ids)

    X_test = np.vstack(X_test)
    ids_test = np.concatenate(ids_test)
    print "Testset:", X_test.shape


    lr_train = []
    lr_test = []
    print "Number of classifiers:", len(classifiers)
    #Now we generate the predictions for each of the classifiers on the training set.
    for indx, classifier in enumerate(classifiers):
        print
        print "Predicting", indx
        t0 = time()
        lr_train.append(classifier.predict(X_train))
        lr_test.append(classifier.predict(X_test))
        print "Done in:", (time() - t0)

    # Convert the predictions into a usable format for the logistic regression.
    lr_train = np.column_stack(lr_train)
    lr_test = np.column_stack(lr_test)

    # Train the logistic regression
    print
    t0 = time()
    param_grid = {'C': [1, 10, 1e2, 1e3, 1e4, 1e5], 'penalty': ['l1', 'l2']}
    clf = GridSearchCV(LogisticRegression(random_state=0), param_grid)
    print "Classifier:"
    print clf
    print "Training."
    clf.fit(lr_train, y_train)
    print "Predicting."
    y_pred = clf.predict(lr_test)
    print "Done in:", (time() - t0)

    print
    filename_submission = "../output/submissionMultiClassStackedGen.csv"
    print "Creating submission file", filename_submission
    f = open(filename_submission, "w")
    print >> f, "Id,Prediction"
    for i in range(len(y_pred)):
        print >> f, str(ids_test[i]) + "," + str(y_pred[i])

    f.close()
    print "Done."