from keras.callbacks import Callback
from keras import backend as K
from math import ceil
from time import ctime
import logging

def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + K.epsilon()) / (sum_ - intersection + K.epsilon())
    return K.mean(jac)

def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + K.epsilon()) / (sum_ - intersection + K.epsilon())
    return K.mean(jac)

def mean_diff(y_true, y_pred):
    return K.mean(y_pred) - K.mean(y_true)

def act_mean(y_true, y_pred):
    return K.mean(y_pred)

def act_min(y_true, y_pred):
    return K.min(y_pred)

def act_max(y_true, y_pred):
    return K.max(y_pred)

def act_std(y_true, y_pred):
    return K.std(y_pred)

def tru_pos(y_true, y_pred):
    axis = K.ndim(y_true) - 1
    ytam = K.argmax(y_true, axis=axis)
    ypam = K.argmax(y_pred, axis=axis)
    prod = ytam * ypam
    return K.sum(prod)

def fls_pos(y_true, y_pred):
    axis = K.ndim(y_true) - 1
    ytam = K.argmax(y_true, axis=axis)
    ypam = K.argmax(y_pred, axis=axis)
    diff = ypam - ytam
    return K.sum(K.clip(diff, 0, 1))

def tru_neg(y_true, y_pred):
    axis = K.ndim(y_true) - 1
    ytam = K.argmin(y_true, axis=axis)
    ypam = K.argmin(y_pred, axis=axis)
    prod = ytam * ypam
    return K.sum(prod)

def fls_neg(y_true, y_pred):
    axis = K.ndim(y_true) - 1
    ytam = K.argmin(y_true, axis=axis)
    ypam = K.argmin(y_pred, axis=axis)
    diff = ypam - ytam
    return K.sum(K.clip(diff, 0, 1))

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_onehot(y_true, y_pred):
    axis = K.ndim(y_true) - 1
    yt_flat = K.cast(K.argmax(y_true, axis=axis), 'float32')
    yp_flat = K.cast(K.argmax(y_pred, axis=axis), 'float32')
    true_positives = K.sum(K.round(K.clip(yt_flat * yp_flat, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(yp_flat, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall_onehot(y_true, y_pred):
    axis = K.ndim(y_true) - 1
    yt_flat = K.cast(K.argmax(y_true, axis=axis), 'float32')
    yp_flat = K.cast(K.argmax(y_pred, axis=axis), 'float32')
    true_positives = K.sum(K.round(K.clip(yt_flat * yp_flat, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(yt_flat, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fmeasure_onehot(y_true, y_pred):
    p = precision_onehot(y_true, y_pred)
    r = recall_onehot(y_true, y_pred)
    return 2 * (p * r) / (p + r + K.epsilon())

def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    return (2.0 * intersection + smooth) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - 1.0 * dice_coef(y_true, y_pred)


