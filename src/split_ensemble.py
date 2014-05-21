__author__ = 'stevelohrenz'
"""Based off of DecMeg2014 example code.

Simple prediction of the class labels of the test set by:
- pooling all the triaining trials of all subjects in one dataset.
- Extracting the MEG data in the first 500ms from when the
  stimulus starts.
- Using a ada boost simple classifier.
"""

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from scipy.io import loadmat


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
    print "Creating the trainset."
    for subject in subjects_train:
        filename = '../data/mat/train_subject%02d.mat' % subject
        print
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

        print
        clf = AdaBoostClassifier(n_estimators=100)
        print "Classifier:"
        print clf
        print "Training.", subject
        clf.fit(np.vstack(XX), yy)
        classifiers.append(clf)

    print
    print "Creating the testset."
    i = 0
    y_pred = []
    subjects_test = range(17, 24)
    for subject in subjects_test:
        filename = '../data/mat/test_subject%02d.mat' % subject
        print
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

        print "Trainset:", np.vstack(XX).shape
        cls_pred = np.zeros(len(XX))

        for indx, clf in enumerate(classifiers):
            print "Predicting."
            cls_pred = np.add(cls_pred, clf.predict(np.vstack(XX)))
        #print cls_pred
        i = i + 1
        # Append the running sum of the predictions to each other and
        y_pred.append(np.vstack(cls_pred))
        ids_test.append(ids)

    y_pred = np.concatenate(y_pred)
    ids_test = np.concatenate(ids_test)
    #print y_pred
    print
    filename_submission = "../output/submissionSplitEnsemble.csv"
    print "Creating submission file", filename_submission
    f = open(filename_submission, "w")
    print >> f, "Id,Prediction"
    for i in range(len(y_pred)):
        print >> f, str(ids_test[i]) + "," + str(int(round(y_pred[i][0] / (len(classifiers)))))

    f.close()
    print "Done."
