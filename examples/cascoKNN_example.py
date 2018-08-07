# -*- coding: utf-8 -*-
"""Example of using kNN for outlier detection
"""
# Author: Chendi Cao <caocd@ksu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.utils import check_X_y
# add two more line before input pyplot
import matplotlib
matplotlib.use('Agg')
# end here 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from pyod.utils.data import get_color_codes
from pyod.utils.data import evaluate_print


def visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
              y_test_pred, show_figure=True,
              save_figure=False):  # pragma: no cover
    """
    Utility function for visualizing the results in examples
    Internal use only

    :param clf_name: The name of the detector
    :type clf_name: str

    :param X_train: The training samples
    :param X_train: numpy array of shape (n_samples, n_features)

    :param y_train: The ground truth of training samples
    :type y_train: list or array of shape (n_samples,)

    :param X_test: The test samples
    :type X_test: numpy array of shape (n_samples, n_features)

    :param y_test: The ground truth of test samples
    :type y_test: list or array of shape (n_samples,)

    :param y_train_pred: The predicted outlier scores on the training samples
    :type y_train_pred: numpy array of shape (n_samples, n_features)

    :param y_test_pred: The predicted outlier scores on the test samples
    :type y_test_pred: numpy array of shape (n_samples, n_features)

    :param show_figure: If set to True, show the figure
    :type show_figure: bool, optional (default=True)

    :param save_figure: If set to True, save the figure to the local
    :type save_figure: bool, optional (default=False)
    """

    if X_train.shape[1] != 2 or X_test.shape[1] != 2:
        raise ValueError("Input data has to be 2-d for visualization. The "
                         "input data has {shape}.".format(shape=X_train.shape))

    X_train, y_train = check_X_y(X_train, y_train)
    X_test, y_test = check_X_y(X_test, y_test)
    c_train = get_color_codes(y_train)
    c_test = get_color_codes(y_test)

    fig = plt.figure(figsize=(12, 10))
    plt.suptitle("Demo of {clf_name}".format(clf_name=clf_name))

    fig.add_subplot(221)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=c_train)
    plt.title('Train ground truth')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='normal',
                              markerfacecolor='b', markersize=8),
                       Line2D([0], [0], marker='o', color='w', label='outlier',
                              markerfacecolor='r', markersize=8)]

    plt.legend(handles=legend_elements, loc=4)

    fig.add_subplot(222)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=c_test)
    plt.title('Test ground truth')
    plt.legend(handles=legend_elements, loc=4)

    fig.add_subplot(223)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred)
    plt.title('Train prediction by {clf_name}'.format(clf_name=clf_name))
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='normal',
                              markerfacecolor='0', markersize=8),
                       Line2D([0], [0], marker='o', color='w', label='outlier',
                              markerfacecolor='yellow', markersize=8)]
    plt.legend(handles=legend_elements, loc=4)

    fig.add_subplot(224)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred)
    plt.title('Test prediction by {clf_name}'.format(clf_name=clf_name))
    plt.legend(handles=legend_elements, loc=4)

    if save_figure:
        plt.savefig('{clf_name}.png'.format(clf_name=clf_name), dpi=300)
    if show_figure:
        plt.show()
    return

# add function to read data function from outside 
# 2018.8.6 Chendi Cao 
def readFile(file,n_train,n_test):
	ntrain = n_train
	ntest = n_test
	n_inliersTest=0
	n_outliersTest=0
	n_inliersTrain=0
	n_outliersTrain=0
	crs = open(file, "r")
	X_trainL, X_testL , y_trainL , y_testL = [],[], [], []
	for columns in ( raw.strip().split() for raw in crs ):  
		if (ntrain>0):
			X_trainL.append([int(columns[0]),int(columns[1])])
			y_trainL.append(int(columns[2]))
			ntrain=ntrain-1
 		if (ntest>0):
			X_testL.append([int(columns[3]),int(columns[4])])
			y_testL.append(int(columns[5]))
			ntest=ntest-1
	X_train = np.array(X_trainL)
	X_test = np.array(X_testL)
	y_train = np.array(y_trainL)
	y_test = np.array(y_testL)
 
	return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    contamination = 0.1  # init=0.1 percentage of outliers
    n_train = 200  # number of training points
    n_test = 100  # number of testing points

#Generate sample data
    # X_train, y_train, X_test, y_test = \
        # generate_data(n_train=n_train,
                      # n_test=n_test,
                      # n_features=2,
                      # contamination=contamination,
                      # random_state=42)

# Get data from CASCO oracle database
# 2017.8.6 Chendi Cao 
    X_train, y_train, X_test, y_test = \
        readFile("data/train30.txt",n_train,n_test )
	
    # train kNN detector
    clf_name = 'KNN'
    clf = KNN()
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    print("\nOn Training Data:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)

    # visualize the results
    visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
              y_test_pred, show_figure=True, save_figure=True)
