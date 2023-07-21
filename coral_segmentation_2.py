from keras.models import load_model
from keras.utils.np_utils import normalize
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, LearningRateScheduler
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

###############################################################################################
# Mask overlay
def segmentation_map_to_image(
    result: np.ndarray, colormap: np.ndarray, remove_holes=False
) -> np.ndarray:
    if len(result.shape) != 2 and result.shape[0] != 1:
        raise ValueError(
            f"Expected result with shape (H,W) or (1,H,W), got result with shape {result.shape}"
        )

    num_classes = colormap.shape[0]
    if len(np.unique(result)) > num_classes:
        raise ValueError(
            f"Expected max {num_classes} classes in result, got {len(np.unique(result))} "
            "different output values. Please make sure to convert the network output to "
            "pixel values before calling this function."
        )
    elif result.shape[0] == 1:
        result = result.squeeze(0)

    result = result.astype(np.uint8)

    contour_mode = cv2.RETR_EXTERNAL if remove_holes else cv2.RETR_TREE
    mask = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
    for label_index, color in enumerate(colormap):
        label_index_map = result == label_index
        label_index_map = label_index_map.astype(np.uint8) * 255
        contours, hierarchies = cv2.findContours(
            label_index_map, contour_mode, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            mask,
            contours,
            contourIdx=-1,
            color=color.tolist(),
            thickness=cv2.FILLED,
        )

    return mask

###################################################################################################################
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # change to CPU
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.833)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
###################################################################################################################
# Paths
image_format = 'tiff'

image_directory = 'D:/Pycharm Projects/Coral Segmentation/raw_data/train/images/'
mask_directory = 'D:/Pycharm Projects/Coral Segmentation/raw_data/train/masks/'

model_folder = 'test_10'
model_save = 'D:/Pycharm Projects/Coral Segmentation/models/' + model_folder

os.makedirs(model_save, exist_ok = True)

model_name = model_save + '/test.hdf5'
logs_name = model_save + '/logs.txt'
graph_name = model_save + '/graph.png'
table_name = model_save + '/table.txt'
###############################################################################################
# Training settings

# 256/ 512
SIZE = 256
# SIZE = 512

# RESIZE = (2048, 1536)
RESIZE = (1536, 1024)
# RESIZE = (1024, 512)

test_size=0.3
batch_size = 8
epochs = 150

remarks = 'Filtered data_variable learning rate'

###############################################################################################
# Data loading
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

# Mask loading with integer encoding
mask_folders = os.listdir(mask_directory)
mask_folders.sort()
# print(masks)

num = 0
for x, mask_folder in enumerate(mask_folders):
    mask_folder = mask_directory + mask_folder + '/tiff_format/'
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

###############################################################################################
# Data curation
image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1), 3)
mask_dataset = np.expand_dims((np.array(mask_dataset)), 3)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=test_size, random_state=42)

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

# One hot encoding
from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

###############################################################################################
# Get model
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

def get_model():
    return multi_unet_model(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

###############################################################################################
# Metrics and Losses
iou_score = sm.metrics.IOUScore(threshold=0.5)
f_score = sm.metrics.FScore(threshold=0.5)
hybrid_metrics = [iou_score, f_score]

def hybrid_loss(y_true, y_pred):
    ce_loss = sm.losses.CategoricalCELoss()(y_true, y_pred)
    # dice_loss = sm.losses.DiceLoss()(y_true, y_pred)
    # focal_loss = sm.losses.CategoricalFocalLoss()(y_true, y_pred)
    # return ce_loss + dice_loss + focal_loss
    # return ce_loss + dice_loss
    return ce_loss
chosen_losses = 'ce'

###############################################################################################
# Learning rate settings
def lr_schedule(epoch):
    initial_lr = 0.001  # Initial learning rate
    decay_rate = 0.1    # Decay rate
    epochs_per_decay = 100  # Number of epochs for each decay

    # New learning rate for current epoch
    lr = initial_lr * (decay_rate ** (epoch // epochs_per_decay))
    if lr < 0.0001:
        lr = 0.0001

    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

class PrintLearningRateCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        print(f"Learning Rate for epoch {epoch + 1}: {lr:.6f}")

###############################################################################################
# Model compilation and callbacks
model = get_model()
model.compile(optimizer='adam', loss=hybrid_loss, metrics=[hybrid_metrics])
model.summary()

#If starting with pre-trained weights. 
# model.load_weights('__.hdf5')

csv_logger =CSVLogger(logs_name, append=True, separator=";")

checkpointer = ModelCheckpoint(model_name, verbose=1, save_best_only=True, mode='min', monitor='val_loss')

callbacks = [csv_logger, checkpointer, lr_scheduler, PrintLearningRateCallback()]

###############################################################################################
# Model training
start = datetime.now()

history = model.fit(X_train, y_train_cat, 
                    batch_size = batch_size, 
                    verbose=1, 
                    epochs=epochs,
                    validation_data=(X_test, y_test_cat), 
                    callbacks=callbacks,
                    #class_weight=class_weights,
                    shuffle=False)

run_time = datetime.now() - start
run_time = run_time.total_seconds() / 60
print("Runtime: {:.2f} minutes".format(run_time))

# model.save('test.hdf5')

############################################################
# Model evaluation
loss_, iou_, f_score = model.evaluate(X_test, y_test_cat)
print("Loss = ", loss_)
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
plt.savefig(graph_name)
plt.show()
##################################
#model = get_model()
# model.load_weights('test.hdf5')  

#IOU
y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

##################################################
# IoU for each class
from keras.metrics import MeanIoU
n_classes = 3
IOU_keras = MeanIoU(num_classes=n_classes)  
print(np.unique(y_test))
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[1,0]+ values[2,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[0,1]+ values[2,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[0,2]+ values[1,2])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)

plt.imshow(image_dataset[0, :,:,0], cmap='gray')
plt.imshow(mask_dataset[0], cmap='gray')
#######################################################################
#Predict on a patched images
#model = get_model()
#model.load_weights('___.hdf5')  
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
# model = load_model(model_name,
#                     custom_objects={"hybrid_loss":hybrid_loss,
#                     "iou_score":iou_score, "f1-score":f_score}, compile=True)

# image = io.imread('D:/Pycharm Projects/Coral Segmentation/test/images/dipsastraea/P8190386.tiff')                     # dips
# image = io.imread('D:/Pycharm Projects/Coral Segmentation/test/images/dipsastraea/P9060348.tiff')                     # dips
image = io.imread('D:/Pycharm Projects/Coral Segmentation/test/images/porites/P9060287.tiff')                         # porites
# image = io.imread('D:/Pycharm Projects/Coral Segmentation/test/images/porites/B-2c_MOV_00_02_47_06_Still097.tiff')    # porites
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = Image.fromarray(image)
large_image = image.resize(RESIZE)
bgr_image = cv2.cvtColor(np.array(large_image), cv2.COLOR_GRAY2BGR)

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

###############################################################################################

height, width = reconstructed_image.shape

colormap = np.array([[0, 0, 0], [0, 255, 0], [0, 0, 255]])

# Define the transparency of the segmentation mask on the photo
alpha = 0.2

# Use function from notebook_utils.py to transform mask to an RGB image
mask = segmentation_map_to_image(reconstructed_image, colormap)

resized_mask = cv2.resize(mask, (width, height))

# Create image with mask put on
image_with_mask1 = cv2.addWeighted(bgr_image, 1-alpha, resized_mask, alpha, 0)
image_with_mask = cv2.cvtColor(image_with_mask1, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.title('Large Image')
plt.imshow(np.array(large_image), cmap='gray')
plt.subplot(222)
plt.title('Prediction of large Image')
plt.imshow(reconstructed_image, cmap='jet')
plt.show()

plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.title('Large Image')
plt.imshow(np.array(large_image), cmap='gray')
plt.subplot(222)
plt.title('Prediction of large Image')
plt.imshow(image_with_mask, cmap='jet')
plt.show()

# green - dipsastarea
# red - porites

#####################################################################
# Result recordings
table = []
col_names = ['Loss', 'IoU', 'F_score','Training Time']
data = [loss_, iou_, f_score, run_time]
table.append(data)
table = np.array(table)

f= open(table_name,"a+")
f.write('Total images: %s\n'%num)
f.write('Test size: %s\n'%test_size)
f.write('Size: %s, %s\n'%(SIZE, RESIZE))
f.write('Batch Size: %s\n'%batch_size)
f.write('Epochs: %s\n'%epochs)
f.write('Losses: %s\n'%chosen_losses)
f.write('Remarks: %s\n'%remarks)
f.write(tabulate(table, headers = col_names))
f.close()