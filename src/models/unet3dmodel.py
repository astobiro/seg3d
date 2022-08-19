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
import pickle
from utils.loss_functions import focal_tversky_loss
from utils.loss_functions import asym_unified_focal_loss
from utils.loss_functions import asymmetric_focal_tversky_loss

class Unet3Dmodel:
    def __init__(self, config):
        K.set_image_data_format("channels_last")
        self.config = Params(config)
        self.resultpath = self.check_results_path()
        # self.dataset = dataset
        self.metrics = [measureDICE]
        self.optim = Adam(self.config.LR)
        # self.preprocess_input = self.preprocess_inputInit()
        self.model = self.modelInit()
        self.loss = self.lossInit(self.config.LOSS)
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
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=1e-9),
            tf.keras.callbacks.CSVLogger(self.resultpath + self.config.TRAINING_OUTPUT_LOG_FILE)
        ]

        return custom_callbacks

    def lossInit(self, loss):
        if loss == "focal_tversky":
            used_loss = focal_tversky_loss()
        elif loss == "asym_focal_tversky":
            used_loss = asymmetric_focal_tversky_loss()
        elif loss == "asym_unified_focal":
            used_loss = asym_unified_focal_loss()
        return used_loss

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
        self.model.compile(optimizer=self.optim, loss=self.loss, metrics=self.metrics) 

        return

    def fit_model(self):
        init = time.time()

        history = self.model.fit(self.train_gen, epochs=self.config.EPOCHS, validation_data = self.val_gen, callbacks = self.callbacks, steps_per_epoch=len(self.train_gen), validation_steps=len(self.val_gen))
        end = time.time()
        print("Training time (secs): {}".format(end-init))
        history = self.model.history

        with open(self.resultpath + "history.pkl", 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        import matplotlib.pyplot as plt
        hist_log_df = history.history

        hist_loss, ax1 = plt.subplots()
        ax1.plot(hist_log_df['loss'],color='b',label='training loss')
        ax1.plot(hist_log_df['val_loss'],color='r',label='validation loss')
        ax1.set_ylabel("loss")
        ax1.set_xlabel("epoch")
        ax1.set_title("Training")
        ax1.legend()
        # plt.show()
        hist_loss.savefig(self.resultpath + self.config.TRAINING_OUTPUT_LOSSGRAPH_FILE)
        plt.close(hist_loss)
        hist_dice, ax2 = plt.subplots()
        ax2.plot(hist_log_df['measureDICE'],color='b',label='training DICE')
        ax2.plot(hist_log_df['val_measureDICE'],color='r',label='validation DICE')
        ax2.set_ylabel("dice")
        ax2.set_xlabel("epoch")
        ax2.set_title("Training")
        ax2.legend()
        # plt.show()
        hist_dice.savefig(self.resultpath + self.config.TRAINING_OUTPUT_DICEGRAPH_FILE)
        plt.close(hist_dice)

        return

    def load_best_results(self):
        self.model.load_weights(self.resultpath + "best_model.h5")
        self.model.compile(self.optim, self.loss, self.metrics)
        print("Loaded weights.")

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
        output_layer = Activation('sigmoid')(fcBlock)
        
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

    def evaluate_model_per_patient(self,n_steps=None,shuffle=False):
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

        pred_list = load_csv_list(self.config.TEST_LIST_FILENAME)
        datagen = self.test_gen
        # pred_gen = VolumeDataGenerator(validation_list_subvolumes, SUBVOLUMES_AXIAL_FOLDER, batch_size=batch_size, dim=target_dim, shuffle=False, verbose=0)
        n_pred = len(datagen)

        if not n_steps is None:
            n_pred = n_steps

        # print(pred_list)
        this_dict = {"IoU": [0], "DICE": [0]}
        # for i in range(len(pred_list)):
        #     pred_list[i] = str(pred_list[i])
        # print("pred_list:",pred_list)
        starting = []
        for i in range(len(self.config.segmentation_name_map)):
            starting.append(str(pred_list[0] + "_" + self.config.segmentation_name_map[i]))
        this_dict = pd.DataFrame(this_dict, index=starting)

        for i in range(1, len(pred_list)):
            starting = []
            for j in range(len(self.config.segmentation_name_map)):
                starting.append(str(pred_list[i] + "_" + self.config.segmentation_name_map[j]))
            y_dict = {"IoU": [0], "DICE": [0]}
            x_dict = pd.DataFrame(y_dict, index=starting)
            this_dict = this_dict.append(x_dict)

        # index ids to numbers
        indexer = res = {val : idx for idx, val in enumerate(pred_list)}
        # print("indexer:", indexer)
        # create dictionary to save values for each id
        ious = np.zeros((len(indexer), len(self.config.segmentation_name_map)))
        dices = np.zeros((len(indexer), len(self.config.segmentation_name_map)))
        # calculate metrics per id
        count = np.zeros(len(indexer))
        for i in range(n_pred):
            batchX, batchy, ID = datagen.getItemWithIDs(i)
            predicted_vals = self.model.predict(batchX)
            # print(len(batchX))
            for j in range(len(batchX)):
                raw_id = ID[j].split("_")[0]
                # print("RAW ID:", raw_id)
                # print("ID", ID)
                pred = predicted_vals[j]
                pred_labels = np.argmax(pred, axis=3)
                for k in range(pred.shape[3]):
                    pred_mask = pred_labels == k
                    gt_mask = batchy[j,:,:,:,k] > 0.5
                    # print(raw_id, indexer[raw_id])
                    ious[indexer[raw_id]][k] += measureIoU(gt_mask, pred_mask)
                    dices[indexer[raw_id]][k] += measureDICE(gt_mask, pred_mask)
                count[indexer[raw_id]]+=1
        # calculate the final metric
        average_metrics = {"IoU": [0,0,0,0,0,0], "DICE": [0,0,0,0,0,0]}
        for i in range(len(pred_list)):
            ious[indexer[pred_list[i]]] = ious[indexer[pred_list[i]]] / count[indexer[pred_list[i]]]
            dices[indexer[pred_list[i]]] = dices[indexer[pred_list[i]]] / count[indexer[pred_list[i]]]
            # put metrics into dataframe
            for j in range(len(self.config.segmentation_name_map)):
                pos = str(pred_list[i] + "_" + self.config.segmentation_name_map[j])
                # float("{0:.4f}".format(x))
                this_dict.loc[pos, "IoU"] = float("{0:.4f}".format(ious[indexer[pred_list[i]]][j]))
                this_dict.loc[pos, "DICE"] = float("{0:.4f}".format(dices[indexer[pred_list[i]]][j]))
                average_metrics["IoU"][j] += float("{0:.4f}".format(ious[indexer[pred_list[i]]][j]))
                average_metrics["DICE"][j] += float("{0:.4f}".format(dices[indexer[pred_list[i]]][j]))
        for i in range(len(average_metrics["IoU"])):
            average_metrics["IoU"][i] = average_metrics["IoU"][i] / len(pred_list)
            average_metrics["DICE"][i] = average_metrics["DICE"][i] / len(pred_list)
        
        this_dict.to_csv(self.resultpath + self.config.EVAL_DF + ".csv")
        
        average_dict = {"IoU": [0], "DICE": [0]}
        average_starting = self.config.segmentation_name_map
        average_dict = pd.DataFrame(average_dict, index=average_starting)
        for i in range(len(average_metrics["IoU"])):
            average_dict.loc[self.config.segmentation_name_map[i], "IoU"] = float("{0:.4f}".format(average_metrics["IoU"][i]))
            average_dict.loc[self.config.segmentation_name_map[i], "DICE"] = float("{0:.4f}".format(average_metrics["DICE"][i]))

        average_dict.to_csv(self.resultpath + self.config.EVAL_DF + "-average.csv")
        print("Metrics saved to files.")

        return

    def save_masks(self):
        subvolumes = []
        datagen = self.test_gen
        model = self.model
        for i in range(len(datagen)):
            x, y, ID = datagen.getItemWithIDs(i)
            pred = model.predict(x)
            for j in range(pred.shape[0]):
                # if ID[j].split("_")[0] != "1.3.6.1.4.1.14519.5.2.1.6279.6001.194465340552956447447896167830":
                #     continue
                batch = pred[j]
                n_classes = batch.shape[3]
                arr_labels = np.argmax(batch, axis=3)
                # print(arr_labels.shape)
                # print(np.unique(arr_labels))
                arr = np.zeros(batch.shape)
                for k in range(n_classes):
                    arr[:,:,:,k] = arr_labels == k
                subvolume = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2]), dtype=np.uint8)
                for l in range(arr.shape[2]):
                    local = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
                    for m in range(n_classes):
                        prob_map = arr[:,:,l,m]
                        # mask = prob_map > 0.5
                        local[prob_map] = self.config.segmentation_labels_map[m]
                    subvolume[:,:,l] = local
                # nib.save(nib.Nifti1Image(subvolume, affine=np.eye(4)), os.path.join(self.resultpath + "predictions/",ID[j]+"_pred.nii.gz"))
                subvolumes.append((subvolume, ID[j]))
        subvolumes_file = open(self.resultpath + "predictions.pkl", 'wb')
        pickle.dump(subvolumes, subvolumes_file)
        subvolumes_file.close()
        print("Predictions saved.")