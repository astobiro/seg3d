from tensorflow.keras.utils import Sequence
import numpy as np
import os
import nibabel as nib

#custom generator
class VolumeDataGenerator(Sequence):
    def __init__(self,
                 name_list,
                 image_folder,
                 batch_size=1,
                 shuffle=True,
                 dim=(160,160,16),
                 label_ids=[],
                 verbose=1,
                 image_suffix = "", 
                 seg_suffix = ""):
        
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
        # IDs = []
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
            # IDs.append(ID)
            
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

    def __data_generation_withIDs(self, list_IDs_temp):
        'Generates data containing batch_size samples'

        # Initialization
        current_batch_size = len(list_IDs_temp)
        
        X = np.zeros((current_batch_size, self.dim[0], self.dim[1], self.dim[2], 1),
                     dtype=np.float32)
        y = np.zeros((current_batch_size, self.dim[0], self.dim[1], self.dim[2], len(self.label_ids)),
                     dtype=np.float32)

        # Generate data
        IDs = []
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
            IDs.append(ID)
            
        return X, y, IDs

    def getItemWithIDs(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[
                  index * self.batch_size: (index + 1) * self.batch_size]
        # Find list of IDs
        name_list_temp = [self.name_list[k] for k in indexes]
        # Generate data
        X, y, ID = self.__data_generation_withIDs(name_list_temp)

        return X, y, ID
    
    def __call__(self):
        for i in self.indexes:
            yield self.__getitem__(i)