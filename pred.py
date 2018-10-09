from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
from keras.models import *
import cv2
import keras



def dice_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
  return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
  return -dice_coef(y_true, y_pred)

pic = load_img('0.png', grayscale=False)
pic = img_to_array(pic)
preds = np.ndarray((1, 299, 299, 3), dtype=np.uint8)
preds[0] = pic
preds = preds.astype('float32')
preds /= 255

model = load_model('model.1400--0.94.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss})
pred = model.predict(preds)
#pred *= 255


f = open('ar.txt', 'w')
for i in range(0, 299):
    str2wtite = ''
    for j in range(0, 299):
        str2wtite += str(pred[0, i, j, 1])
    f.write(str2wtite)
f.close()


#print(pred.shape)
full = pred[0, :, :, 0]
print(full.shape)
cv2.imshow('0', full)
#cv2.imwrite('0chan.jpg', chans[6])
cv2.waitKey(0)