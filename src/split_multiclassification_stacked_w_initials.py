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
import time


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
    Xp_train = []
    y_train = []
    X_test = []
    Xp_test = []
    ids_test = []

    classifiers = []

    total_start_time = time.time()
    print
    print "Creating the trainset. And training the classifiers for each subject. "
    for subject in subjects_train:
        filename = '../data/mat/train_subject%02d.mat' % subject
        print
        print "Loading", filename
        data = loadmat(filename, squeeze_me=True)
        XXX = data['X']
        yy = data['y']
        sfreq = data['sfreq']
        tmin_original = data['tmin']
        print "Dataset summary:"
        print "XX:", XXX.shape
        print "yy:", yy.shape
        print "sfreq:", sfreq

        XX = create_features(XXX, tmin, tmax, sfreq)
        Xp = create_features(XXX, -0.5, 0.0, sfreq)

        X_train.append(XX)
        Xp_train.append(Xp)
        y_train.append(yy)

        xx = np.vstack(XX)

        '''start_time = time.time()
        print
        adaClf = AdaBoostClassifier(n_estimators=60)
        print "Classifier:"
        print adaClf
        print "Training", subject
        adaClf.fit(xx, yy)
        classifiers.append(adaClf)
        print "Time:", time.time() - start_time'''

        '''start_time = time.time()
        print
        rfClf = RandomForestClassifier(n_estimators=20, n_jobs=-1)
        print "Classifier:"
        print rfClf
        print "Training", subject
        rfClf.fit(xx, yy)
        classifiers.append(rfClf)
        print "Time:", time.time() - start_time

        start_time = time.time()
        print
        sgdClf = SGDClassifier(loss="hinge", penalty="l2")
        print "Classifier:"
        print sgdClf
        print "Training", subject
        sgdClf.fit(xx, yy)
        classifiers.append(sgdClf)
        print "Time:", time.time() - start_time

        start_time = time.time()
        print
        nnClf = KNeighborsClassifier(n_neighbors=7, algorithm='ball_tree', weights='distance')
        print "Classifier:"
        print nnClf
        print "Training", subject
        nnClf.fit(xx, yy)
        classifiers.append(nnClf)
        print "Time:", time.time() - start_time '''

        start_time = time.time()
        print
        lrClf = LogisticRegression(C=1, penalty='l2', random_state=0)
        print "Classifier:"
        print lrClf
        print "Training", subject
        lrClf.fit(xx, yy)
        classifiers.append(lrClf)
        print "Time:", time.time() - start_time

    # X_train is the full stack of subjects
    X_train = np.vstack(X_train)
    Xp_train = np.vstack(Xp_train)
    y_train = np.concatenate(y_train)
    print "Trainset:", X_train.shape

    print
    print "Creating the testset."
    subjects_test = range(17, 24)
    for subject in subjects_test:
        print
        filename = '../data/mat/test_subject%02d.mat' % subject
        print "Loading", filename
        data = loadmat(filename, squeeze_me=True)
        XXX = data['X']
        ids = data['Id']
        sfreq = data['sfreq']
        tmin_original = data['tmin']
        print "Dataset summary:"
        print "XX:", XXX.shape
        print "ids:", ids.shape
        print "sfreq:", sfreq

        XX = create_features(XXX, tmin, tmax, sfreq)
        Xp = create_features(XXX, -0.5, 0.0, sfreq)

        X_test.append(XX)
        Xp_test.append(Xp)
        ids_test.append(ids)

    X_test = np.vstack(X_test)
    Xp_test = np.vstack(Xp_test)
    ids_test = np.concatenate(ids_test)
    print "Testset:", X_test.shape


    lr_train = []
    lr_test = []
    #Now we generate the predictions for each of the classifiers on the training set.
    print "Number classifiers: ", len(classifiers)
    for indx, classifier in enumerate(classifiers):
        print "Predicting ", indx
        print classifier
        lr_train.append(classifier.predict(X_train))
        lr_test.append(classifier.predict(X_test))

    # Convert the predictions into a usable format for the logistic regression.
    lr_train = np.column_stack(lr_train)
    lr_test = np.column_stack(lr_test)

    # Add in xp_train and Xp_test to the lr_train, lr_test arrays.
    lr_train = [np.concatenate((lr_train[i], Xp_train[i])) for i in range(len(lr_train))]
    lr_test = [np.concatenate((lr_test[i], Xp_test[i])) for i in range(len(lr_test))]

    # Train the logistic regression
    start_time = time.time()
    print
    clf = LogisticRegression(C=1, penalty='l2', random_state=0) # Beware! You need 10Gb RAM to train LogisticRegression on all 16 subjects!
    print "Classifier:"
    print clf
    print "Training."
    clf.fit(lr_train, y_train)
    print "Predicting."
    y_pred = clf.predict(lr_test)
    print "LR Time:", time.time() - start_time

    print
    filename_submission = "../output/submissionMultiClassStackedGenBase.csv"
    print "Creating submission file", filename_submission
    f = open(filename_submission, "w")
    print >> f, "Id,Prediction"
    for i in range(len(y_pred)):
        print >> f, str(ids_test[i]) + "," + str(y_pred[i])

    f.close()
    print "Done in: ", time.time() - total_start_time