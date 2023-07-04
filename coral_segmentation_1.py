from keras.models import load_model
from keras.utils.np_utils import normalize
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
# from keras.optimizers import adam_v2
import tensorflow as tf
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pickle
from datetime import datetime
from tabulate import tabulate
from simple_multi_unet_model import multi_unet_model
import segmentation_models as sm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import class_weight
from skimage import io
from patchify import patchify

###################################################################################################################
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # change to CPU
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.833)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
###################################################################################################################

image_format = 'tiff'

image_directory = 'D:/Pycharm Projects/Coral Segmentation/train/images/'
mask_directory = 'D:/Pycharm Projects/Coral Segmentation/train/masks/'
model_name = 'test_3.hdf5'

###############################################################################################
# data preparation

# 256/ 512
# SIZE = 256
SIZE = 512

# RESIZE = (2048, 1536)
RESIZE = (1536, 1024)

image_dataset = [] 
mask_dataset = []  

image_folders = os.listdir(image_directory)
image_folders.sort()
# print(images)

num = 0
for x, image_folder in enumerate(image_folders):
    image_folder = image_directory + image_folder + '/tiff_format/'
    images = os.listdir(image_folder)
    images.sort()
    for i, image_name in enumerate(images):  
        if (image_name.split('.')[-1] == image_format):
            # print(image_directory+image_name)
            image = io.imread(image_folder + image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = Image.fromarray(image)
            image = image.resize(RESIZE)
            patches_img = patchify(np.array(image), (SIZE, SIZE), step=SIZE)
            for j in range(patches_img.shape[0]):
                for k in range(patches_img.shape[1]):
                    single_patch_img = patches_img[j, k, :, :]
                    image_dataset.append(np.array(single_patch_img))
                    num = num + 1
print("number of images:\n", num)

mask_folders = os.listdir(mask_directory)
mask_folders.sort()
# print(masks)

num = 0
for x, mask_folder in enumerate(mask_folders):
    mask_folder = mask_directory + mask_folder + '/'
    masks = os.listdir(mask_folder)
    masks.sort()
    for i, mask_name in enumerate(masks):  
        if (mask_name.split('.')[-1] == image_format):
            # print(mask_directory+mask_name)
            mask = io.imread(mask_folder + mask_name)
            mask = Image.fromarray(mask)
            mask = mask.resize(RESIZE)
            patches_msk = patchify(np.array(mask), (SIZE, SIZE), step=SIZE)
            for j in range(patches_msk.shape[0]):
                for k in range(patches_msk.shape[1]):
                    single_patch_msk = patches_msk[j, k, :, :]
                    label = x+1
                    single_patch_msk = np.where(np.array(single_patch_msk) > 0, label, single_patch_msk)
                    single_patch_msk = np.where(np.array(single_patch_msk) < 0, 0, single_patch_msk)
                    mask_dataset.append(np.array(single_patch_msk))
                    num = num + 1
print("number of masks:\n", num)


image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1), 3)
mask_dataset = np.expand_dims((np.array(mask_dataset)), 3)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.20, random_state=32)

import random
for x in range(10):
    image_number = random.randint(0, len(X_train)-1)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(np.reshape(X_train[image_number], (SIZE, SIZE)), cmap='gray')
    plt.subplot(122)
    mask_ = np.reshape(y_train[image_number], (SIZE, SIZE))
    plt.imshow(mask_)
    plt.title('%s : %s' % (image_number, np.unique(mask_)))
    plt.show()

print(np.unique(y_train))
n_classes = len(np.unique(y_train))

from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

###############################################################

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

iou_score = sm.metrics.IOUScore(threshold=0.5)
f_score = sm.metrics.FScore(threshold=0.5)
hybrid_metrics = [iou_score, f_score]

def hybrid_loss(y_true, y_pred):
    ce_loss = sm.losses.CategoricalCELoss()(y_true, y_pred)
    dice_loss = sm.losses.DiceLoss()(y_true, y_pred)
    focal_loss = sm.losses.CategoricalFocalLoss()(y_true, y_pred)
    return ce_loss + dice_loss + focal_loss

model = get_model()
model.compile(optimizer='adam', loss=hybrid_loss, metrics=[hybrid_metrics])
model.summary()

#If starting with pre-trained weights. 
#model.load_weights('__.hdf5')

csv_logger =CSVLogger('logs.txt', append=True, separator=";")

checkpointer = ModelCheckpoint(model_name, verbose=1, save_best_only=True, mode='min', monitor='val_loss')

callbacks = [csv_logger, checkpointer]

history = model.fit(X_train, y_train_cat, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=100,
                    validation_data=(X_test, y_test_cat), 
                    callbacks=callbacks,
                    #class_weight=class_weights,
                    shuffle=False)

# model.save('test.hdf5')

############################################################
# evaluate model
loss_, iou_, f_score = model.evaluate(X_test, y_test_cat)
print("Loss = ", (loss_ * 100.0), "%")
print("IoU = ", iou_)
print("FScore = ", f_score)


 # Plot training & validation iou_score values
plt.figure(figsize=(30, 5))
 
plt.subplot(131)
plt.plot(history.history['f1-score'])
plt.plot(history.history['val_f1-score'])
plt.title('Model fscore')
plt.ylabel('fscore')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.ylim(0, 1)

plt.subplot(132)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.ylim(0, 1)

# Plot training & validation loss values
plt.subplot(133)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.ylim(0, 1)
plt.show()
#######################################################################
#Predict on a patched images
#model = get_model()
#model.load_weights('???.hdf5')  
import random
test_img_number = random.randint(0, len(X_test)-1)
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()

#####################################################################
#Predict on large image (test images)
from patchify import patchify, unpatchify

# image = io.imread('D:/Pycharm Projects/Coral Segmentation/test/images/dipsastraea/P8190386.tiff')                     # dips
# image = io.imread('D:/Pycharm Projects/Coral Segmentation/test/images/dipsastraea/P9060348.tiff')                     # dips
image = io.imread('D:/Pycharm Projects/Coral Segmentation/test/images/porites/P9060287.tiff')                         # porites
# image = io.imread('D:/Pycharm Projects/Coral Segmentation/test/images/porites/B-2c_MOV_00_02_47_06_Still097.tiff')    # porites
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = Image.fromarray(image)
large_image = image.resize(RESIZE)

patches = patchify(np.array(large_image), (SIZE, SIZE), step=SIZE)

predicted_patches = []
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        print(i,j)
        
        single_patch = patches[i,j,:,:]       
        single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
        single_patch_input=np.expand_dims(single_patch_norm, 0)
        single_patch_prediction = (model.predict(single_patch_input))
        single_patch_predicted_img=np.argmax(single_patch_prediction, axis=3)[0,:,:]

        predicted_patches.append(single_patch_predicted_img)

predicted_patches = np.array(predicted_patches)

predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], SIZE,SIZE) )

reconstructed_image = unpatchify(predicted_patches_reshaped, np.array(large_image).shape)

plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.title('Large Image')
plt.imshow(np.array(large_image), cmap='gray')
plt.subplot(222)
plt.title('Prediction of large Image')
plt.imshow(reconstructed_image, cmap='jet')
plt.show()

# green - dipsastarea
# red - porites