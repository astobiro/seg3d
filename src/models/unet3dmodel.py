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
from tensorflow.keras.models import load_model
import skimage.io as io
import nrrd
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv3D, PReLU, UpSampling3D, concatenate , Reshape, Dense, Permute, MaxPool3D
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation, add, GaussianNoise, BatchNormalization, multiply
import pandas as pd
import re
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
#from loss import custom_loss
from utils.utils import Params
from utils.utils import load_csv_list
from utils.utils import get_full_filenames
from utils.utils import custom_loss
from utils.utils import measureDICE
from pprint import pprint
import tensorflow as tf
from generators.data_loader import VolumeDataGenerator

class Unet3Dmodel:
    def __init__(self, config):
        K.set_image_data_format("channels_last")
        self.config = Params(config)
        self.resultpath = self.check_results_path()
        # self.dataset = dataset
        # self.metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
        self.optim = Adam(self.config.LR)
        # self.preprocess_input = self.preprocess_inputInit()
        self.model = self.modelInit()
        # self.dice_loss = sm.losses.DiceLoss()
        # self.focal_loss = self.focal_lossInit()
        self.total_loss = self.total_lossInit()
        self.callbacks = self.callbacksInit()
        self.train_gen, self.val_gen, self.test_gen = self.initGenerators()
        return

    def check_results_path(self):
        resultpath = "data/results/" + self.config.TESTNO + "/"
        if not os.path.exists(resultpath):
            os.mkdir(resultpath)
        return resultpath

    def modelInit(self):
        input_shape = tuple(self.config.INPUT_SHAPE)
        model = self.my_unet3D(input_shape = input_shape, output_channels = 6)
        self.model_to_df(model)
        return model

    def callbacksInit(self):
        custom_callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15),
            tf.keras.callbacks.ModelCheckpoint(self.resultpath + self.config.TRAINING_OUTPUT_MODEL_FILE, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=False),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-9),
            tf.keras.callbacks.CSVLogger(self.resultpath + self.config.TRAINING_OUTPUT_LOG_FILE)
        ]

        return custom_callbacks

    def total_lossInit(self):
        total_loss = custom_loss
        return total_loss

    def initGenerators(self):
        training_list_subvolumes = load_csv_list(self.config.TRAINING_LIST_SUBVOLUMES_AXIAL_FILENAME)
        validation_list_subvolumes = load_csv_list(self.config.VALIDATION_LIST_SUBVOLUMES_AXIAL_FILENAME)
        test_list_subvolumes = load_csv_list(self.config.TEST_LIST_SUBVOLUMES_AXIAL_FILENAME)

        label_ids = self.config.segmentation_labels_map
        img_suff = self.config.IMAGE_SUFFIX
        seg_suff = self.config.SEG_SUFFIX
        batch_size = self.config.BATCH_SIZE
        target_dim = tuple(self.config.TARGET_DIM)

        train_gen = VolumeDataGenerator(training_list_subvolumes, self.config.SUBVOLUMES_AXIAL_FOLDER, batch_size=batch_size, dim=target_dim, verbose=0, label_ids = label_ids, image_suffix = img_suff, seg_suffix = seg_suff)
        val_gen = VolumeDataGenerator(validation_list_subvolumes, self.config.SUBVOLUMES_AXIAL_FOLDER, batch_size=batch_size, dim=target_dim, shuffle=False, verbose=0, label_ids = label_ids, image_suffix = img_suff, seg_suffix = seg_suff)
        test_gen = VolumeDataGenerator(test_list_subvolumes, self.config.SUBVOLUMES_AXIAL_FOLDER, batch_size=batch_size, dim=target_dim, shuffle=False, verbose=0, label_ids = label_ids, image_suffix = img_suff, seg_suffix = seg_suff)

        return train_gen, val_gen, test_gen

    def define_model(self):
        learning_rate = self.config.LR
        loss_function = self.total_loss

        target_dim = tuple(self.config.TARGET_DIM)
        batch_size = self.config.BATCH_SIZE

        sgd = SGD(learning_rate=learning_rate, momentum=0.9, decay=0)
        self.model.compile(optimizer=sgd, loss=loss_function, metrics=[measureDICE]) 

        return

    def fit_model(self):
        init = time.time()
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
          try:
            for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
          except RuntimeError as e:
            print(e)
        history = self.model.fit(self.train_gen, epochs=self.config.EPOCHS, validation_data = self.val_gen, callbacks = self.callbacks, steps_per_epoch=len(self.train_gen), validation_steps=len(self.val_gen))
        end = time.time()
        print("Training time (secs): {}".format(end-init))
        history = self.model.history

        import matplotlib.pyplot as plt
        hist_log_df = history

        plt.plot(hist_log_df['loss'],color='b',label='training loss')
        plt.plot(hist_log_df['val_loss'],color='r',label='validation loss')
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("Training")
        plt.legend()
        #plt.show()
        plt.savefig(self.resultpath + self.config.TRAINING_OUTPUT_LOSSGRAPH_FILE)

        plt.plot(hist_log_df['measureDICE'],color='b',label='training DICE')
        plt.plot(hist_log_df['val_measureDICE'],color='r',label='validation DICE')
        plt.ylabel("dice")
        plt.xlabel("epoch")
        plt.title("Training")
        plt.legend()
        #plt.show()
        plt.savefig(self.resultpath + self.config.TRAINING_OUTPUT_DICEGRAPH_FILE)

        return

    def my_unet3D(self, input_shape=(160,160,16,4),output_channels=4):
        
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

    def model_to_df(self, model):
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
        df.to_csv(self.resultpath + self.config.MODEL_SUMMARY_DF_FILE)

        return

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

    def evaluate_model_per_patient(self,n_steps=None,shuffle=False):
        pred_list = load_csv_list(self.config.TEST_LIST_SUBVOLUMES_AXIAL_FILENAME)
        datagen = self.test_gen
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
            predicted_vals = self.model.predict(batchX)
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
        
        this_dict.to_csv(self.resultpath + self.config.EVAL_DF)
        
        return

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