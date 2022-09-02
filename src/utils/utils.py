import matplotlib.pyplot as plt
import bz2
import pickle
import _pickle as cPickle
import json
import csv
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow as tf

import sys
SRC_ABSOLUTE_PATH = "."
sys.path.append(SRC_ABSOLUTE_PATH)

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

def measureDICE(y_true, y_pred):
    y_true    = K.reshape(y_true,shape=(-1,4))
    y_pred    = K.reshape(y_pred,shape=(-1,4))
    print(y_true.shape)
    print(y_pred.shape)
    err = 10e-9
    intersection = tf.reduce_sum(y_true * y_pred)
    dice_score = (2.0 * K.sum(intersection) + err) / (K.sum(y_true) + K.sum(y_pred) + err)
    return dice_score

def visualize(image):
    """PLot images in one row."""
    plt.figure(figsize=(16, 5))
    plt.imshow(image)
    plt.show()

def plot_stuff(images, masks, pos=0, color_map = 'nipy_spectral'):
    '''
    Plots and a slice with all available annotations
    '''
    fig = plt.figure(figsize=(18,15))

    plt.subplot(1,4,1)
    plt.imshow(images[pos].squeeze(), cmap='bone')
    plt.title('Original Image')

    plt.subplot(1,4,2)
    plt.imshow(images[pos].squeeze(), cmap='bone')
    plt.imshow(masks[pos].squeeze(), alpha=0.5, cmap=color_map)
    plt.title('Lung Mask')

# generates a pickled object
def full_pickle(title, data):
    pikd = open(title + '.pickle', 'wb')
    pickle.dump(data, pikd)
    pikd.close()

# loads and returns a pickled objects
def loosen(file):
    pikd = open(file, 'rb')
    data = pickle.load(pikd)
    pikd.close()
    return data

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.RandomCrop(height=320, width=320, always_apply=True),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(384, 480)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)