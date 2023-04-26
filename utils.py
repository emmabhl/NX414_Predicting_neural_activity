### Utils
import h5py
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

def load_it_data(path_to_data):
    """ Load IT data

    Args:
        path_to_data (str): Path to the data

    Returns:
        np.array (x6): Stimulus train/val/test; objects list train/val/test; spikes train/val
    """

    datafile = h5py.File(os.path.join(path_to_data,'IT_data.h5'), 'r')

    stimulus_train = datafile['stimulus_train'][()]
    spikes_train = datafile['spikes_train'][()]
    objects_train = datafile['object_train'][()]
    
    stimulus_val = datafile['stimulus_val'][()]
    spikes_val = datafile['spikes_val'][()]
    objects_val = datafile['object_val'][()]
    
    stimulus_test = datafile['stimulus_test'][()]
    objects_test = datafile['object_test'][()]

    ### Decode back object type to latin
    objects_train = [obj_tmp.decode("latin-1") for obj_tmp in objects_train]
    objects_val = [obj_tmp.decode("latin-1") for obj_tmp in objects_val]
    objects_test = [obj_tmp.decode("latin-1") for obj_tmp in objects_test]

    return stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val


def visualize_img(stimulus,objects,stim_idx):
    """Visualize image given the stimulus and corresponding index and the object name.

    Args:
        stimulus (array of float): Stimulus containing all the images
        objects (list of str): Object list containing all the names
        stim_idx (int): Index of the stimulus to plot
    """    
    normalize_mean=[0.485, 0.456, 0.406]
    normalize_std=[0.229, 0.224, 0.225]

    img_tmp = np.transpose(stimulus[stim_idx],[1,2,0])

    ### Go back from normalization
    img_tmp = (img_tmp*normalize_std + normalize_mean) * 255

    plt.figure()
    plt.imshow(img_tmp.astype(np.uint8),cmap='gray')
    plt.title(str(objects[stim_idx]))
    plt.show()
    return

def best_alpha_Ridge(X, y, alphas):
    """implement cross validation to find the best alpha for Ridge regression

    Args:
        X (ndarray): input data
        y (ndarray): output data, neuronal activity
        alphas (list of double): list of alpha to test

    Returns:
        tuple (double, ndarray): best alpha and all the scores for each alpha
    """
    scores = []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        cv_scores = cross_val_score(model, X, y, cv=5)
        scores.append(np.mean(cv_scores))
    return alphas[np.argmax(scores)], scores

def plot_RidgeCV(alphas, scores):
    """plot the scores for each alpha

    Args:
        alphas (list of double): list of alpha that were tested
        scores (list of double): list of scores for each alpha
    """
    plt.figure(figsize=(3,2))
    plt.plot(alphas, scores)
    plt.xlabel('alpha')
    plt.ylabel('r2 score')
    plt.show()
    
def RidgeCV(X, y, alphas):
    """find the best alpha for Ridge regression and plot the scores for each alpha, then fit the model with the best alpha

    Args:
        X (ndarray): input data
        y (ndarray): output data, neuronal activity
        alphas (list of double): list of alpha to test

    Returns:
        tuple (model, double): the ridge model fitted with the best alpha and the corresponding alpha
    """
    best_alpha, scores = best_alpha_Ridge(X, y, alphas)
    plot_RidgeCV(alphas, scores)
    print('The best alpha is', best_alpha)
    model = Ridge(alpha=best_alpha)
    model.fit(X, y)
    return model, best_alpha