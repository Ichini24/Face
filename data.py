import glob
import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import re
import os

class dataProcess(object):
	def __init__(self, out_rows, out_cols, data_path='D://300W//full//preproced//respics//', label_path='D://300W//full//preproced//dstL//',
				 test_path="test", npy_path="npydata", img_type="png"):
		self.out_rows = out_rows
		self.out_cols = out_cols
		self.data_path = data_path
		self.label_path = label_path
		self.img_type = img_type
		self.test_path = test_path
		self.npy_path = npy_path


	def __get_one_out(self, counter):
		#ar = np.ndarray((self.out_rows, self.out_cols, 7), dtype=np.uint8)
		picar = glob.glob(self.label_path + str(counter) + "//*.png")
		pics = np.ndarray((5, self.out_rows, self.out_cols), dtype=np.uint8)
		for i in range(0, 5):
			pics[i] = cv2.imread(picar[i], 0)
		ret = np.stack((pics[0], pics[1], pics[2], pics[3], pics[4]), axis=2)
		return ret



	def create_train_data(self):
		i = 0
		imgdatas = np.ndarray((399, self.out_rows, self.out_cols, 3), dtype=np.uint8)
		imglabels = np.ndarray((399, self.out_rows, self.out_cols, 5), dtype=np.uint8)

		for i in range(0, 399):

			img = load_img(self.data_path + str(i) + '.png', grayscale=False)
			img = img_to_array(img)
			imgdatas[i] = img

			lbl = self.__get_one_out(i)
			imglabels[i] = lbl
		print('loading done')
		np.save(self.npy_path + '/imgs_train.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
		print('Saving to .npy files done.')

	def load_train_data(self):
		imgs_train = np.load(self.npy_path + "/imgs_train.npy")
		imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 255
		# mean = imgs_train.mean(axis = 0)
		# imgs_train -= mean
		imgs_mask_train /= 255
		return imgs_train, imgs_mask_train
