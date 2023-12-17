from ..util import util
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from pathlib import Path
import numpy as np
from torch import tensor
from ..util.preprocess import preprocess_image
import warnings

class CustomImageDataset(Dataset):
    """
    The purpose of this class is to load input and target image pairs from two separate directories, 
    preprocess the input image if required, and return them as a tuple of PyTorch tensors.    
    """
    def __init__(self, list_folders, enable_preprocess = False):
        valid_suffix = {".tif", ".tiff"}
        self.enable_preprocess = enable_preprocess        
        self.data_augmentation_flag = False
        self.list_folders = list_folders
        #reading all files in target folder
        
        #reading all files in target folder
        self.input_images_path = []
        for folder_path in list_folders:
            current_image_paths = [p for p in Path(folder_path).iterdir() if p.suffix in valid_suffix] 
            self.input_images_path += current_image_paths

        #check that every image has the corresponding csv file with target coordinates
        check_trainingset_file_matching(self.input_images_path)

        #check that every csv file has the same size
        self.target_shape = check_csv_size_matching(self.input_images_path)
        
        #check images all the images have commun shape
        check_training_set_equal_img_sizes(self.input_images_path)
        
        self.num_classes = len(self.list_folders)
        
    def __len__(self):
        return len(self.input_images_path)
    
    def __targetshape__(self):
        return self.target_shape
        
    def __getitem__(self, idx):
        #reading input and target images
        input_img = util.imread(self.input_images_path[idx])
        target = util.pandas_read_array(Path(self.input_images_path[idx].parent, self.input_images_path[idx].stem + '.csv') )
        
        #preprocess image if required
        if self.enable_preprocess:
            input_img = preprocess_image(input_img)
            
        #converting numpy to tensor
        input_img = tensor(input_img.astype(np.float32)).float()
        target = tensor(target).float()
        
        #converting targets to one_hot encoding. [1,C]. Loss functions requires [C,1]
        # target = one_hot(target, num_classes= self.num_classes).permute(1,0)

        #Pytorch CNN requires dimensions to be [C,W,H] for images. Make sure that we have Channel dimension
        input_img = input_img.unsqueeze(0) if input_img.dim() == 2 else input_img
            
        if self.data_augmentation_flag:
            input_img = self.data_augmentation_object.run(input_img)
        
        return input_img, target


    def set_data_augmentation(self, augmentation_flag = False, data_augmentation_object = None):
        """
        this method is used to set a data augmentation flag and object. 
        The data_augmentation_flag is a boolean indicating whether data augmentation should be performed or not
        data_augmentation_object is an object containing the data augmentation methods to be applied.
        """
        self.data_augmentation_flag = augmentation_flag
        self.data_augmentation_object = data_augmentation_object
        if not(augmentation_flag):
            self.data_augmentation_object = None
        
def check_trainingset_file_matching(input_images_path):
    #verify each image in input images has an associated csv file
    missing_files = [f for f in input_images_path if not Path(f.parent, f.stem + '.csv').is_file()]
    if missing_files:
        raise ValueError('Missing csv files for images: ' + ', '.join(missing_files))
    
def check_csv_size_matching(input_images_path):
    # we require the output to have the same number of coordinates
    target_0 = util.pandas_read_array(Path(input_images_path[0].parent, input_images_path[0].stem + '.csv') )
    for file_path in input_images_path:
        current_csv_target = util.pandas_read_array(Path(file_path.parent, file_path.stem + '.csv'))
        if not target_0.shape == current_csv_target.shape:
            raise ValueError(f"CSV file {input_images_path[0]} does not have the same shape as {file_path}")
    return target_0        

def check_training_set_equal_img_sizes(input_images_path):
    #verify that every image has the same shape
    img_shape = util.imread(input_images_path[0]).shape
    
    for file_path in input_images_path:
        current_img_shape = util.imread(file_path).shape
        if current_img_shape!=img_shape:
            raise ValueError("Training set have images with different shape. Images file {input_images_path[0]} does not have the same shape as {file_path}")
    print("\nPass: all images have the same shape")