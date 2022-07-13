import os
import csv
import time
import random
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import nibabel as nib

from skimage import io
from skimage.transform import rescale, resize
from skimage import img_as_uint, img_as_ubyte, exposure
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Activation, Dense, GlobalAveragePooling2D, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, Input, concatenate, Cropping2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import load_model
import skimage.io as io
import nrrd
from keras_buoy.models import ResumableModel
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv3D, PReLU, UpSampling3D, concatenate , Reshape, Dense, Permute, MaxPool3D
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation, add, GaussianNoise, BatchNormalization, multiply
from tensorflow.keras.optimizers import SGD
import pandas as pd
import re
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

subvolumeno = 1

IMAGE_FOLDER = "LUNA16/LUNA16_LA"
SEG_FOLDER = "LUNA16/annotations"
IMAGE_LIST_FILENAME = "LUNA16/name_mapping.csv"
TRAINING_LIST_FILENAME = "LUNA16/" + TESTFOLDER + "training.csv"
VALIDATION_LIST_FILENAME = "LUNA16/" + TESTFOLDER + "validation.csv"
TEST_LIST_FILENAME = "LUNA16/" + TESTFOLDER + "test.csv"

if subvolumeno == 1:
    TRAINING_LIST_SUBVOLUMES_AXIAL_FILENAME = "LUNA16/" + TESTFOLDER + "training_subvolumes_axial.csv"
    VALIDATION_LIST_SUBVOLUMES_AXIAL_FILENAME = "LUNA16/" + TESTFOLDER + "validation_subvolumes_axial.csv"
    TEST_LIST_SUBVOLUMES_AXIAL_FILENAME = "LUNA16/" + TESTFOLDER + "test_subvolumes_axial.csv"

    SUBVOLUMES_AXIAL_FOLDER = "LUNA16/subvolumes/axial_test"
else:
    TRAINING_LIST_SUBVOLUMES_AXIAL_FILENAME = "LUNA16/training_subvolumes_axial_100.csv"
    VALIDATION_LIST_SUBVOLUMES_AXIAL_FILENAME = "LUNA16/validation_subvolumes_axial_100.csv"
    TEST_LIST_SUBVOLUMES_AXIAL_FILENAME = "LUNA16/test_subvolumes_axial_100.csv"

    SUBVOLUMES_AXIAL_FOLDER = "LUNA16/subvolumes/axial_100"

IMAGE_SUFFIX = "raw"
SEG_SUFFIX = "seg"

MODEL_SUMMARY_DF_FILE = "LUNA16/" + TESTFOLDER + "model_summary.csv"
#Arquivos usados para treinamento
TRAINING_OUTPUT_MODEL_FILE = "LUNA16/" + TESTFOLDER + "out_model_3D_current-{loss:.4f}.h5"
TRAINING_OUTPUT_LOG_FILE = "LUNA16/" + TESTFOLDER + "out_log_3D_current.csv"
TRAINING_OUTPUT_LOSSGRAPH_FILE = "LUNA16/" + TESTFOLDER + "lossgraph_current.png"
TRAINING_OUTPUT_DICEGRAPH_FILE = "LUNA16/" + TESTFOLDER + "dicegraph_current.png"
TRAINING_MODEL_CONTINUE_FILE = "LUNA16/" + TESTFOLDER + "model-continue.h5"

PRE_TRAINED_MODEL_FILE = "LUNA16/" + TESTFOLDER + "out_model_3D.h5"
PRE_TRAINED_LOG_FILE = "LUNA16/" + TESTFOLDER + "out_log_3D.csv"
PRE_TRAINED_LOSSGRAPH_FILE = "LUNA16/" + TESTFOLDER + "lossgraph.png"
PRE_TRAINED_DICEGRAPH_FILE = "LUNA16/" + TESTFOLDER + "dicegraph.png"

segmentation_name_map = ["Background", "SuperiorLeft", "SuperiorRight", "Middle", "LowerRight", "LowerLeft"]
segmentation_color_map = [(0,0,0.5),(144/255,238/255,144/255),(110/255,184/255,209/255),(219/255,82/255,66/255),(221/255,130/255,101/255),(174/255,72/255,58/255)]
segmentation_labels_map = [0, 7, 4, 5, 6, 8]

def get_full_filenames(name,image_folder,seg_folder):
    image_full_filename = os.path.join(image_folder,name+".mhd")
    # print(name)
    seg_full_filename = os.path.join(seg_folder,name+"_LobeSegmentation.nrrd")
    #print(seg_full_filename)
    return image_full_filename, seg_full_filename

def load_csv_list(csv_filename):
    out_list = []
    with open(csv_filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            out_list.append(row[0])
    return out_list


image_df = pd.read_csv(IMAGE_LIST_FILENAME)
# image_df.head()

training_list = load_csv_list(TRAINING_LIST_FILENAME)
validation_list = load_csv_list(VALIDATION_LIST_FILENAME)
test_list = load_csv_list(TEST_LIST_FILENAME)

training_df = image_df[image_df['ID'].isin(training_list)]
training_names = list(training_df['ID'])

validation_df = image_df[image_df['ID'].isin(validation_list)]
validation_names = list(validation_df['ID'])

test_df = image_df[image_df['ID'].isin(test_list)]
test_names = list(test_df['ID'])
print(len(test_names))

#custom generator
class VolumeDataGenerator(Sequence):
    def __init__(self,
                 name_list,
                 image_folder,
                 batch_size=1,
                 shuffle=True,
                 dim=(160,160,16),
                 label_ids=segmentation_labels_map,
                 verbose=1,
                 image_suffix = IMAGE_SUFFIX, 
                 seg_suffix = SEG_SUFFIX):
        
        self.name_list = name_list
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dim = dim
        self.label_ids = label_ids
        self.verbose = verbose
        self.image_suffix = image_suffix
        self.seg_suffix = seg_suffix
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.name_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.name_list) / self.batch_size))

    def normalize_image(self, image):
        b = np.percentile(image, 99)
        t = np.percentile(image, 1)
        image = np.clip(image, t, b)
        if np.std(image)==0:
            return image
        else:
            image = (image - np.mean(image)) / np.std(image)
            return image
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'

        # Initialization
        current_batch_size = len(list_IDs_temp)
        
        X = np.zeros((current_batch_size, self.dim[0], self.dim[1], self.dim[2], 1),
                     dtype=np.float32)
        y = np.zeros((current_batch_size, self.dim[0], self.dim[1], self.dim[2], len(self.label_ids)),
                     dtype=np.float32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if self.verbose == 1:
                print("Processing: %s" % ID)
                
            image_filename = os.path.join(self.image_folder, ID+"_"+self.image_suffix+".nii.gz")
            seg_filename = os.path.join(self.image_folder, ID+"_"+self.seg_suffix+".nii.gz")
            
            image = np.array(nib.load(image_filename).get_fdata(), dtype=np.float32)
            seg_image = np.array(nib.load(seg_filename).get_fdata(), dtype=np.float32)
            
            
            if seg_image.shape[0]!=self.dim[0] or seg_image.shape[1]!=self.dim[1]:
                #TODO: implement resize
                pass
            
            X[i,:,:,:,0] = self.normalize_image(image)
            
            for j, label_id in enumerate(self.label_ids):
                y[i,:,:,:,j] = (seg_image == label_id).astype(np.float32)
            
        return X, y

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[
                  index * self.batch_size: (index + 1) * self.batch_size]
        # Find list of IDs
        name_list_temp = [self.name_list[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(name_list_temp)

        return X, y
    
    def __call__(self):
        for i in self.indexes:
            yield self.__getitem__(i)


training_list_subvolumes = load_csv_list(TRAINING_LIST_SUBVOLUMES_AXIAL_FILENAME)
validation_list_subvolumes = load_csv_list(VALIDATION_LIST_SUBVOLUMES_AXIAL_FILENAME)
test_list_subvolumes = load_csv_list(TEST_LIST_SUBVOLUMES_AXIAL_FILENAME)

target_dim = (160,160,16)

batch_size = 5

training_generator = VolumeDataGenerator(training_list_subvolumes, SUBVOLUMES_AXIAL_FOLDER, batch_size=batch_size, dim=target_dim, verbose=0)
validation_generator = VolumeDataGenerator(validation_list_subvolumes, SUBVOLUMES_AXIAL_FOLDER, batch_size=batch_size, dim=target_dim, shuffle=False, verbose=1)
test_generator = VolumeDataGenerator(test_list_subvolumes, SUBVOLUMES_AXIAL_FOLDER, batch_size=batch_size, dim=target_dim, shuffle=False, verbose=0)

def create_vol_summary_image(arr, n_columns = 4):
    n_sub_images = arr.shape[2] 
    h = arr.shape[0]
    w = arr.shape[1]
    target_rows = int(np.ceil(n_sub_images/n_columns))
    collage_h = target_rows*h
    collage_w = w*n_columns
    collage = np.zeros((collage_h,collage_w))
    for i in range(n_sub_images):
        offset_h = (i//n_columns)*h
        offset_w = (i%n_columns)*w
        collage[offset_h:(offset_h+h),offset_w:(offset_w+w)] = arr[:,:,i]
    return collage

def create_vol_seg_summary_image(in_arr, n_columns = 4, colors=segmentation_color_map, use_argmax = True):
    n_classes = in_arr.shape[3]
    if use_argmax:
        arr_labels = np.argmax(in_arr, axis=3)
        arr = np.zeros(in_arr.shape, dtype=np.float32)
        for k in range(n_classes):
            arr[:,:,:,k] = (arr_labels == k).astype(np.float32)
    else:
        arr = in_arr
    n_sub_images = arr.shape[2] 
    h = arr.shape[0]
    w = arr.shape[1]
    
    target_rows = int(np.ceil(n_sub_images/n_columns))
    collage_h = target_rows*h
    collage_w = w*n_columns
    collage = np.zeros((collage_h,collage_w,3),dtype=np.uint8)
    for i in range(n_sub_images):
        offset_h = (i//n_columns)*h
        offset_w = (i%n_columns)*w
        local_rgb = np.zeros((h,w,3),dtype=np.uint8)
        for k in range(n_classes):
            prob_map = arr[:,:,i,k]
            mask = prob_map>0.5
            color = segmentation_color_map[k]
            local_rgb[mask]=[color[0]*255.0, color[1]*255.0, color[2]*255.0]
        collage[offset_h:(offset_h+h),offset_w:(offset_w+w)] = local_rgb
    return collage


def show_iterator_images(iterator,sampled_batches_n=2,label_ids=segmentation_labels_map,colors=segmentation_color_map):
    n_rows = (sampled_batches_n*iterator.batch_size)
    
    fig, axes = plt.subplots(n_rows,2,figsize=(5,n_rows),dpi=400)
    row = 0
    for i in range(sampled_batches_n):
        batchX, batchy = iterator.__getitem__(i)
        print('Batch X shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
        print('Batch y shape=%s, min=%.3f, max=%.3f' % (batchy.shape, batchy.min(), batchy.max()))
        for j in range(batchy.shape[0]):
            image_flair = batchX[j,:,:,:,0].copy()
            
            collage_flair = create_vol_summary_image(image_flair)

            axes[row][0].imshow(collage_flair,cmap='gray')
            
            collage_rgb_seg = create_vol_seg_summary_image(batchy[j,:,:,:,:])
            
            axes[row][1].imshow(collage_rgb_seg)

            axes[row][0].set_axis_off()
            
            row += 1    
    iterator.on_epoch_end()

def generalized_dice(y_true, y_pred):
    
    """
    Generalized Dice Score
    https://arxiv.org/pdf/1707.03237
    
    """
    
    y_true    = K.reshape(y_true,shape=(-1,4))
    y_pred    = K.reshape(y_pred,shape=(-1,4))
    sum_p     = K.sum(y_pred, -2)
    sum_r     = K.sum(y_true, -2)
    sum_pr    = K.sum(y_true * y_pred, -2)
    weights   = K.pow(K.square(sum_r) + K.epsilon(), -1)
    generalized_dice = (2 * K.sum(weights * sum_pr)) / (K.sum(weights * (sum_r + sum_p)))
    
    return generalized_dice

def generalized_dice_loss(y_true, y_pred):   
    return 1-generalized_dice(y_true, y_pred)
    
    
def custom_loss(y_true, y_pred):
    
    """
    The final loss function consists of the summation of two losses "GDL" and "CE"
    with a regularization term.
    """
    
    return generalized_dice_loss(y_true, y_pred) + 1.25 * categorical_crossentropy(y_true, y_pred)


import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv3D, PReLU, UpSampling3D, concatenate , Reshape, Dense, Permute, MaxPool3D
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation, add, GaussianNoise, BatchNormalization, multiply
from tensorflow.keras.optimizers import SGD
#from loss import custom_loss
K.set_image_data_format("channels_last")


def my_unet3D(input_shape=(160,160,16,4),output_channels=4):
    
    input_layer = Input(shape=input_shape, name='the_input_layer')
    
    encoderBlock32 = Conv3D(32, 3, padding='same')(input_layer)
    encoderBlock32 = Activation("relu")(encoderBlock32)
    encoderBlock32 = Conv3D(32, 3, padding='same')(encoderBlock32)
    encoderBlock32 = Activation("relu")(encoderBlock32)
    
    encoderBlock64 = MaxPool3D(pool_size=(2, 2, 2))(encoderBlock32)
    encoderBlock64 = Conv3D(64, 3, padding='same')(encoderBlock64)
    encoderBlock64 = Activation("relu")(encoderBlock64)
    encoderBlock64 = Conv3D(64, 3, padding='same')(encoderBlock64)
    encoderBlock64 = Activation("relu")(encoderBlock64)
    
    encoderBlock128 = MaxPool3D(pool_size=(2, 2, 2))(encoderBlock64)
    encoderBlock128 = Conv3D(128, 3, padding='same')(encoderBlock128)
    encoderBlock128 = Activation("relu")(encoderBlock128)
    encoderBlock128 = Conv3D(128, 3, padding='same')(encoderBlock128)
    encoderBlock128 = Activation("relu")(encoderBlock128)
    
    encoderBlock256 = MaxPool3D(pool_size=(2, 2, 2))(encoderBlock128)
    encoderBlock256 = Conv3D(256, 3, padding='same')(encoderBlock256)
    encoderBlock256 = Activation("relu")(encoderBlock256)
    encoderBlock256 = Conv3D(256, 3, padding='same')(encoderBlock256)
    encoderBlock256 = Activation("relu")(encoderBlock256)
    
    decoderBlock128 = UpSampling3D(size=(2, 2, 2))(encoderBlock256)
    decoderBlock128 = Conv3D(128, 2, padding='same')(decoderBlock128)
    decoderMerge128 = concatenate([encoderBlock128,decoderBlock128]) #axis = -1
    decoderBlock128 = Conv3D(128, 3, padding='same')(decoderMerge128)
    decoderBlock128 = Activation("relu")(decoderBlock128)
    decoderBlock128 = Conv3D(128, 3, padding='same')(decoderBlock128)
    decoderBlock128 = Activation("relu")(decoderBlock128)
    
    decoderBlock64 = UpSampling3D(size=(2, 2, 2))(decoderBlock128)
    decoderBlock64 = Conv3D(64, 2, padding='same')(decoderBlock64)
    decoderMerge64 = concatenate([encoderBlock64,decoderBlock64]) #axis = -1
    decoderBlock64 = Conv3D(64, 3, padding='same')(decoderMerge64)
    decoderBlock64 = Activation("relu")(decoderBlock64)
    decoderBlock64 = Conv3D(64, 3, padding='same')(decoderBlock64)
    decoderBlock64 = Activation("relu")(decoderBlock64)
    
    decoderBlock32 = UpSampling3D(size=(2, 2, 2))(decoderBlock64)
    decoderBlock32 = Conv3D(32, 2, padding='same')(decoderBlock32)
    decoderMerge32 = concatenate([encoderBlock32,decoderBlock32]) #axis = -1
    decoderBlock32 = Conv3D(32, 3, padding='same')(decoderMerge32)
    decoderBlock32 = Activation("relu")(decoderBlock32)
    decoderBlock32 = Conv3D(32, 3, padding='same')(decoderBlock32)
    decoderBlock32 = Activation("relu")(decoderBlock32)
    
    fcBlock = Conv3D(output_channels, 1, padding='same')(decoderBlock32)
    output_layer = Activation('softmax')(fcBlock)
    
    out_model = Model(inputs = input_layer, outputs = output_layer)
    
    return out_model

input_shape = (160,160,16,1)

try:
    del(model)
except:
    print("Can not delete model")
    
model = my_unet3D(input_shape = input_shape, output_channels=6)
model.summary()

import pandas as pd
import re
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

def model_to_df(model):
    def clean(x):
        return x.replace('[0][0]','').replace('(','').replace(')','').replace('[','').replace(']','')

    def magic(x):
        tmp = re.split(r'\s{1,}',clean(x))
        return tmp

    width = 250
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    stringlist = stringlist[2:]
    for i in range(len(stringlist)):
        stringlist[i] = stringlist[i].lstrip()
        stringlist[i] = re.split(r'\s\s+', stringlist[i])
    for i in range(2, len(stringlist)-5, 3):
        # print(i)
        stringlist[i][1] = stringlist[i][1] + stringlist[i+1][0]
        # print(stringlist[i])
    newstring = []
    header = stringlist[0][:-1]
    for i in range(2, len(stringlist)-5, 3):
        newstring.append(stringlist[i][:-1])
    newstring.append([stringlist[-4], "", 0, ""])
    newstring.append([stringlist[-3], "", 0, ""])
    newstring.append([stringlist[-2], "", 0, ""])
    # print(newstring)
    df = pd.DataFrame(columns=header)
    for index,entry in enumerate(newstring):
        # print(index, entry)
        df.loc[index] = {header[0] : entry[0],
                        header[1] : tuple([int(e) for e in clean(entry[1]).split(', ')[1:]]),
                        header[2] : int(entry[2]),
                        header[3] : [clean(e) for e in entry[3:]]}
    #df['layername'],df['type'] = zip(*df['Layer (type)'].map(magic))
    #df['kernels'] = [e[-1:][0] for e in df['Output Shape']]
    return df

df = model_to_df(model)

df.to_csv(MODEL_SUMMARY_DF_FILE)

def measureDICE(y_true, y_pred):
    y_true    = K.reshape(y_true,shape=(-1,4))
    y_pred    = K.reshape(y_pred,shape=(-1,4))
    err = 10e-9
    intersection = tf.reduce_sum(y_true * y_pred)
    dice_score = (2.0 * K.sum(intersection) + err) / (K.sum(y_true) + K.sum(y_pred) + err)
    return dice_score

learning_rate = 0.001
loss_function = custom_loss

target_dim = (160,160,16)
batch_size = 5

train_gen = VolumeDataGenerator(training_list_subvolumes, SUBVOLUMES_AXIAL_FOLDER, batch_size=batch_size, dim=target_dim, verbose=0)
val_gen = VolumeDataGenerator(validation_list_subvolumes, SUBVOLUMES_AXIAL_FOLDER, batch_size=batch_size, dim=target_dim, shuffle=False, verbose=0)
test_gen = VolumeDataGenerator(test_list_subvolumes, SUBVOLUMES_AXIAL_FOLDER, batch_size=batch_size, dim=target_dim, shuffle=False, verbose=0)

sgd = SGD(learning_rate=learning_rate, momentum=0.9, decay=0)
model.compile(optimizer=sgd, loss=loss_function, metrics=[measureDICE])

custom_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15),
    tf.keras.callbacks.ModelCheckpoint(TRAINING_OUTPUT_MODEL_FILE, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=False),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-9),
    tf.keras.callbacks.CSVLogger(TRAINING_OUTPUT_LOG_FILE)
]
model = ResumableModel(model, custom_objects={'custom_loss':custom_loss, 'measureDICE':measureDICE}, save_every_epochs=1, to_path=TRAINING_MODEL_CONTINUE_FILE)

init = time.time()
history = model.fit(train_gen, epochs=50, validation_data = val_gen, callbacks = custom_callbacks, steps_per_epoch=len(train_gen), validation_steps=len(val_gen))
end = time.time()
print("Training time (secs): {}".format(end-init))

history = model.history

import matplotlib.pyplot as plt
hist_log_df = model.history

plt.plot(hist_log_df['loss'],color='b',label='training loss')
plt.plot(hist_log_df['val_loss'],color='r',label='validation loss')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training")
plt.legend()
#plt.show()
plt.savefig(TRAINING_OUTPUT_LOSSGRAPH_FILE)

plt.plot(hist_log_df['measureDICE'],color='b',label='training DICE')
plt.plot(hist_log_df['val_measureDICE'],color='r',label='validation DICE')
plt.ylabel("dice")
plt.xlabel("epoch")
plt.title("Training")
plt.legend()
#plt.show()
plt.savefig(TRAINING_OUTPUT_DICEGRAPH_FILE)

def measureIoU(gt_mask, prediction_mask, err = 10e-9):
    intersection = np.logical_and(gt_mask, prediction_mask)
    union = np.logical_or(gt_mask, prediction_mask)
    iou_score = (np.sum(intersection) + err) / (np.sum(union) + err)
    return iou_score

def measureDICE(gt_mask, prediction_mask, err = 10e-9):
    intersection = np.logical_and(gt_mask, prediction_mask)
    dice_score = (2.0 * np.sum(intersection) + err) / (np.sum(gt_mask) + np.sum(prediction_mask) + err)
    return dice_score

def measureSensitivity(gt_mask, prediction_mask, err = 10e-9):
    tp = np.sum(np.logical_and(gt_mask, prediction_mask))
    tn = np.sum(np.logical_and(~gt_mask, ~prediction_mask))
    fp = np.sum(np.logical_and(~gt_mask, prediction_mask))
    fn = np.sum(np.logical_and(gt_mask, ~prediction_mask))
    sensitivity = tp/(tp+fn+err)
    return sensitivity

def measureSpecifity(gt_mask, prediction_mask, err = 10e-9):
    tp = np.sum(np.logical_and(gt_mask, prediction_mask))
    tn = np.sum(np.logical_and(~gt_mask, ~prediction_mask))
    fp = np.sum(np.logical_and(~gt_mask, prediction_mask))
    fn = np.sum(np.logical_and(gt_mask, ~prediction_mask))
    specificity = tn/(tn+fp+err)
    return specificity

def evaluate_model_per_patient(_model,datagen,ids,n_steps=None,shuffle=False):
    pred_list = ids
    # pred_gen = VolumeDataGenerator(validation_list_subvolumes, SUBVOLUMES_AXIAL_FOLDER, batch_size=batch_size, dim=target_dim, shuffle=False, verbose=0)
    n_pred = len(datagen)

    if not n_steps is None:
        n_pred = n_steps

    this_dict = {"IoU": [0], "DICE": [0], "Sensitivity": [0], "Specificity": [0]}
    for i in range(len(pred_list)):
        pred_list[i] = str(pred_list[i])
    starting = []
    for i in range(len(segmentation_name_map)):
        starting.append(str(pred_list[0] + "_" + segmentation_name_map[i]))
    this_dict = pd.DataFrame(this_dict, index=starting)

    for i in range(1, len(pred_list)):
        starting = []
        for j in range(len(segmentation_name_map)):
            starting.append(str(pred_list[i] + "_" + segmentation_name_map[j]))
        y_dict = {"IoU": [0], "DICE": [0], "Sensitivity": [0], "Specificity": [0]}
        x_dict = pd.DataFrame(y_dict, index=starting)
        this_dict = this_dict.append(x_dict)

    # index ids to numbers
    indexer = res = {val : idx for idx, val in enumerate(pred_list)}
    # create dictionary to save values for each id
    ious = np.zeros((len(indexer), len(segmentation_name_map)))
    dices = np.zeros((len(indexer), len(segmentation_name_map)))
    sensitivities = np.zeros((len(indexer), len(segmentation_name_map)))
    specifities = np.zeros((len(indexer), len(segmentation_name_map)))
    # calculate metrics per id
    count = np.zeros(len(indexer))
    for i in range(n_pred):
        batchX, batchy, ID = datagen.__getitem__(i)
        predicted_vals = _model.predict(batchX)
        for j in range(len(batchX)):
            raw_id = ID[j].split("_")[0]
            pred = predicted_vals[j]
            pred_labels = np.argmax(pred, axis=3)
            for k in range(pred.shape[3]):
                pred_mask = pred_labels == k
                gt_mask = batchy[j,:,:,:,k] > 0.5
                ious[indexer[raw_id]][k] += measureIoU(gt_mask, pred_mask)
                dices[indexer[raw_id]][k] += measureDICE(gt_mask, pred_mask)
                sensitivities[indexer[raw_id]][k] += measureSensitivity(gt_mask, pred_mask)
                specifities[indexer[raw_id]][k] += measureSpecifity(gt_mask, pred_mask)
            count[indexer[raw_id]]+=1
    # calculate the final metric
    for i in range(len(pred_list)):
        ious[indexer[pred_list[i]]] = ious[indexer[pred_list[i]]] / count[indexer[pred_list[i]]]
        dices[indexer[pred_list[i]]] = dices[indexer[pred_list[i]]] / count[indexer[pred_list[i]]]
        sensitivities[indexer[pred_list[i]]] = sensitivities[indexer[pred_list[i]]] / count[indexer[pred_list[i]]]
        specifities[indexer[pred_list[i]]] = specifities[indexer[pred_list[i]]] / count[indexer[pred_list[i]]]
        # put metrics into dataframe
        for j in range(len(segmentation_name_map)):
            pos = str(pred_list[i] + "_" + segmentation_name_map[j])
            # float("{0:.4f}".format(x))
            this_dict.loc[pos, "IoU"] = float("{0:.4f}".format(ious[indexer[pred_list[i]]][j]))
            this_dict.loc[pos, "DICE"] = float("{0:.4f}".format(dices[indexer[pred_list[i]]][j]))
            this_dict.loc[pos, "Sensitivity"] = float("{0:.4f}".format(sensitivities[indexer[pred_list[i]]][j]))
            this_dict.loc[pos, "Specificity"] = float("{0:.4f}".format(specifities[indexer[pred_list[i]]][j]))
    
    return this_dict

def save_masks(val_gen, model):
	# temp = val_gen.__getitem__(0)
	# valid_pred_subvolumes = np.zeros((len(val_gen), 5, 160, 160, 16), dtype=np.uint8)
	# print(valid_pred_subvolumes.shape)
	for i in range(len(val_gen)):
	    x, y, ID = val_gen.__getitem__(i)
	    pred = model.model.predict(x)
	    # batch_subvolumes = np.zeros((pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]), dtype=np.uint8)
	    for j in range(pred.shape[0]):
	        bigX = x[j,:,:,:,:]
	        bigY = y[j,:,:,:,:]
	        batch = pred[j,:,:,:,:]
	        n_classes_X = bigX.shape[3]
	        n_classes = batch.shape[3]
	        n_classes_Y = bigY.shape[3]
	        arr_labels_X = np.argmax(bigX, axis=3)
	        arr_labels = np.argmax(batch, axis=3)
	        arr_labels_Y = np.argmax(bigY, axis=3)
	        arr_X = np.zeros(bigX.shape, dtype=np.float32)
	        arr = np.zeros(batch.shape, dtype=np.float32)
	        arr_Y = np.zeros(batch.shape, dtype=np.float32)
	        for k in range(n_classes):
	            arr[:,:,:,k] = (arr_labels == k).astype(np.float32)
	        for k in range(n_classes_X):
	            arr_X[:,:,:,k] = (arr_labels_X == k).astype(np.float32)
	        for k in range(n_classes_Y):
	            arr_Y[:,:,:,k] = (arr_labels_Y == k).astype(np.float32)
	        subvolume_X = np.zeros((arr_X.shape[0], arr_X.shape[1], arr_X.shape[2]), dtype=np.uint8)
	        subvolume = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2]), dtype=np.uint8)
	        subvolume_Y = np.zeros((arr_Y.shape[0], arr_Y.shape[1], arr_Y.shape[2]), dtype=np.uint8)
	        for l in range(arr.shape[2]):
	            local = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
	            local_Y = np.zeros((arr_Y.shape[0], arr_Y.shape[1]), dtype=np.uint8)
	            for m in range(n_classes):
	                prob_map = arr[:,:,l,m]
	                prob_map_Y = arr_Y[:,:,l,m]
	                mask = prob_map > 0.5
	                mask_Y = prob_map > 0.5
	                # print(local_X.shape, mask_X.shape)
	                local[mask] = segmentation_labels_map[m]
	                local_Y[mask_Y] = segmentation_labels_map[m]
	                # print(np.unique(local_X, return_counts=True))
	            subvolume[:,:,l] = local
	            subvolume_X[:,:,l] = bigX[:,:,l,0]
	            subvolume_Y[:,:,l] = local_Y
	            # nib.save(nib.Nifti1Image(subvolume_X, affine=np.eye(4)), os.path.join("LUNA16/results/predictions",ID[j]+"_raw.nii.gz"))
	            nib.save(nib.Nifti1Image(subvolume, affine=np.eye(4)), os.path.join("LUNA16/" + TESTFOLDER + "predictions",ID[j]+"_pred.nii.gz"))
	            # nib.save(nib.Nifti1Image(subvolume_Y, affine=np.eye(4)), os.path.join("LUNA16/results/test1/predictions",ID[j]+"_gt.nii.gz"))
	            # print(subvolume_X.shape)
	            # print(np.unique(subvolume_X, return_counts=True))
	            # input()
	        # batch_subvolumes[j,:,:,:] = subvolume
	    # valid_pred_subvolumes[i,:,:,:,:] = batch_subvolumes