import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from termcolor import colored


def matrix_f1_score(y_true, y_pred):
    """
    Execute f1 scores for SRL
    :param y_true: matrix
    :param y_pred: matrix
    :return:
    """

    fp, tp, fn = 0, 0, 0

    for true_mat, pred_mat in zip(y_true, y_pred):
        for true_row, pred_row in zip(true_mat, pred_mat):
            tp += sum(1 for t, p in zip(true_row, pred_row) if p == t and t != 0)
            fp += sum(1 for t, p in zip(true_row, pred_row) if p != t and p != 0)
            fn += sum(1 for t, p in zip(true_row, pred_row) if p != t and p == 0 and t != 0)

    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        precision = 0
        recall = 0
        f1 = 0

    print(colored(f"precision: {precision}, recall: {recall}, f1_score:{f1}", color="green"))
    return precision, recall, f1


def f1_score(y_true, y_pred, to_print=True):
    correct_asw = 0.0
    given_asw = 0.0
    total_asw = 0.0

    correct_asw += len([1 for p, t in zip(y_pred, y_true) if p == t])
    given_asw += len([elem for elem in y_pred if elem])
    total_asw += len(y_pred)

    recall = correct_asw / total_asw
    try:
        precision = correct_asw / given_asw
    except ZeroDivisionError:
        precision = 0
    try:
        f1 = 2 * recall * precision / (recall + precision)
    except ZeroDivisionError:
        f1 = 0
    if to_print:
        print(colored(f"precision: {precision}, recall: {recall}, f1_score:{f1}", color="green"))

    return f1, recall, precision


def normalize_tensors(base, to_norm):
    base = tf.to_float(base)
    to_norm = tf.to_float(to_norm)

    concat = tf.stack([base, to_norm], axis=0)

    concat = tf.div(
        tf.subtract(
            concat,
            tf.reduce_min(concat)
        ),
        tf.subtract(
            tf.reduce_max(concat),
            tf.reduce_min(concat)
        )
    )

    return concat[1]


def get_dict_value(dictionary, search_key, error_return):
    try:
        return dictionary[search_key]
    except KeyError:
        return error_return


def confusion_matrix_plot(y_true, y_pred, output_classes, base_name, path=None, show=False):
    """
    Plot and save confusion matrix into specified dir
    """
    # get paths

    idx = len([elem for elem in os.listdir(path) if base_name in elem])
    file = os.path.join(path, f"{base_name}_{idx}.png")

    cm = np.zeros(shape=(len(output_classes), len(output_classes)))
    for true_mat, pred_mat in zip(y_true, y_pred):
        for true_row, pred_row in zip(true_mat, pred_mat):
            for true, pred in zip(true_row, pred_row):
                if true != pred:
                    cm[true][pred] += 1

    # create figure and popolate
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + output_classes, rotation=45)
    ax.set_yticklabels([''] + output_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # save image
    plt.savefig(file)

    if show:
        plt.show()

    plt.close()


def video_from_cms(path):
    """
    Create a video from the confusion matrix images
    :return:
    """
    import cv2

    print("Saving cm videos")

    max_sec_gif = 20

    train_cm = [os.path.join(path, elem) for elem in os.listdir(path)]

    train_cm.sort(key=natural_keys)

    print(f"Number of frames in test, train :{len(train_cm)}")

    train_frames = cv2.imread(train_cm[0])

    h, w, _ = train_frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    fps = 5

    skip_every_train = math.ceil(len(train_cm) / (max_sec_gif * fps))

    del train_cm[skip_every_train - 1::skip_every_train]

    # train_cm=train_cm[:fps*max_sec_gif]
    # test_cm=test_cm[:fps*max_sec_gif]

    print(f"Number of frames in test, train after skipping:{len(train_cm)}")

    save_path = os.path.join(path, "train.avi")

    train_video = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    for idx in range(len(train_cm)):
        train_video.write(cv2.imread(train_cm[idx]))

    cv2.destroyAllWindows()
    train_video.release()

    print("Cm videos saved")


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''

    def atoi(text):
        return int(text) if text.isdigit() else text

    return [atoi(c) for c in re.split('(\d+)', text)]


def root(num, r=2):
    """
    Execute the n-th root of a number
    :param num: the number for the root
    :param r: the n-th
    :return:
    """
    return np.roots([1] + [0] * (r - 1) + [-num])
