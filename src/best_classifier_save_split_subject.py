"""

Using dimensionality reduction to find the best test features.
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
from sklearn.externals import joblib

'''
Takes the data set and takes out only the part you want to see.
'''
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

'''
Find the best classifier for the given data set.
Will save to disk the classifier after found.
Assume that the data sets will be over subjects or over entire set.
if coming back into this method then it will load the classifier from file instead of recomputing. '''
def find_classifier(XX, yy, subject, model, best=True):

    print
    print '*********************************************'
    print 'Finding classifier for %s subject, %s ' % (subject, model)
    filename = '../data/classifiers/%sSubject%02d%s.clf'
    print XX.shape
    if best:
        name_addendum = 'Best' + str(XX.size)
    else:
        name_addendum = 'Default' + str(XX.size)

    # block for finding the LogisticRegression.
    if model == 'LogisticRegression' or model == 'All':
        lr_filename = filename % ('LogisticRegression', subject, name_addendum)
        try:
            lr_clf = joblib.load(lr_filename)
        except IOError:
            print "No job is available for %s " % lr_filename

            print
            t0 = time()
            if best:
                param_grid = {'C': [1, 10, 50, 1e2, 5e2], 'penalty': ['l1', 'l2']}
                lr_clf = GridSearchCV(LogisticRegression(random_state=0), param_grid)
            else:
                lr_clf = LogisticRegression(random_state=0)
            print "Classifier:"
            print lr_clf
            print "Training", subject
            lr_clf.fit(xx, yy)
            # dump the classifier
            joblib.dump(lr_clf, lr_filename, compress=9)
            print "Done in:", (time() - t0)
        print
        print "Best estimator found by grid search:"
        print lr_clf.best_estimator_
        return lr_clf

    if model == 'AdaBoostClassifier' or model == 'All':
        lr_filename = filename % ('AdaBoostClassifier', subject, name_addendum)
        try:
            lr_clf = joblib.load(lr_filename)
        except IOError:
            print "No job is available for %s " % lr_filename

            print
            t0 = time()
            if best:
                param_grid = {'learning_rate': [1, 5], 'n_estimators': [150, 200, 500]}
                lr_clf = GridSearchCV(AdaBoostClassifier(random_state=0), param_grid)
            else:
                lr_clf = AdaBoostClassifier(random_state=0)
            print "Classifier:"
            print lr_clf
            print "Training", subject
            lr_clf.fit(xx, yy)
            # dump the classifier
            joblib.dump(lr_clf, lr_filename, compress=9)
            print "Done in:", (time() - t0)
        print
        print 'Best estimator found by grid search for %s subject, %s' % (subject, model)
        print lr_clf.best_estimator_
        print '*********************************************'
        print
        return lr_clf

    if model == 'RandomForestClassifier' or model == 'All':
        lr_filename = filename % ('RandomForestClassifier', subject, name_addendum)
        try:
            lr_clf = joblib.load(lr_filename)
        except IOError:
            print "No job is available for %s " % lr_filename

            print
            t0 = time()
            if best:
                param_grid = {'max_features': ['auto', None], 'n_estimators': [200, 500, 750, 1000]}
                lr_clf = GridSearchCV(RandomForestClassifier(max_depth=None, n_jobs=-1, random_state=0), param_grid)
            else:
                lr_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
            print "Classifier:"
            print lr_clf
            print "Training", subject
            lr_clf.fit(xx, yy)
            # dump the classifier
            joblib.dump(lr_clf, lr_filename, compress=9)
            print "Done in:", (time() - t0)
        print
        print "Best estimator found by grid search:"
        print lr_clf.best_estimator_
        return lr_clf

    if model == 'SGDClassifier' or model == 'All':
        lr_filename = filename % ('SGDClassifier', subject, name_addendum)
        try:
            lr_clf = joblib.load(lr_filename)
        except IOError:
            print "No job is available for %s " % lr_filename

            print
            t0 = time()
            if best:
                param_grid = {'loss': ['hinge', 'squared_hinge', 'perceptron'], 'penalty': ['l1', 'l2']}
                lr_clf = GridSearchCV(SGDClassifier(random_state=0), param_grid)
            else:
                lr_clf = SGDClassifier(random_state=0)
            print "Classifier:"
            print lr_clf
            print "Training", subject
            lr_clf.fit(xx, yy)
            # dump the classifier
            joblib.dump(lr_clf, lr_filename, compress=9)
            print "Done in:", (time() - t0)
        print
        print "Best estimator found by grid search:"
        print lr_clf.best_estimator_
        return lr_clf

    if model == 'KNeighborsClassifier' or model == 'All':
        lr_filename = filename % ('KNeighborsClassifier', subject, name_addendum)
        try:
            lr_clf = joblib.load(lr_filename)
        except IOError:
            print "No job is available for %s " % lr_filename

            print
            t0 = time()
            if best:
                param_grid = {'n_neighbors': [3, 5, 10, 15, 20]}
                lr_clf = GridSearchCV(KNeighborsClassifier(weights='uniform', algorithm='ball_tree', random_state=0), param_grid)
            else:
                lr_clf = KNeighborsClassifier(random_state=0)
            print "Classifier:"
            print lr_clf
            print "Training", subject
            lr_clf.fit(xx, yy)
            # dump the classifier
            joblib.dump(lr_clf, lr_filename, compress=9)
            print "Done in:", (time() - t0)
        print
        print "Best estimator found by grid search:"
        print lr_clf.best_estimator_
        return lr_clf


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

        # This is the list of classifiers we want to build into our grid.
        listClassifiers = ['RandomForestClassifier', 'KNeighborsClassifier', 'LogisticRegression', 'AdaBoostClassifier']

        # Iterate over the list of classifiers and find the classifier for this subject, append that classifier to classifiers.
        for classifier in listClassifiers:
            lrClf = find_classifier(xx, yy, subject, classifier, best=True)
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

    # Train the logistic regression L2 - In practice, could change out very easily for another classifier once
    # all the best subject classifiers are trained.
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