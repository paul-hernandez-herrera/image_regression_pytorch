from torchvision.transforms import Resize
import torchvision.transforms.v2 as transforms
from torchvision import datapoints
import numpy as np
from torch import tensor

class augmentation_task():
    def __init__(self, zoom_range = [0.8, 1.2],
                 shear_angle = [-5, 5], 
                 img_resize = [224,224],
                 rotation_angle = [0,0],
                 enable_shear = True, 
                 enable_hflip = True, 
                 enable_vflip = True, 
                 enable_zoom = True,
                 enable_rotation = False,
                 enable_resize = False):
        self.zoom_range = zoom_range
        self.shear_angle = shear_angle
        self.rotation_angle = rotation_angle
        
        #flag to compute specific transformations
        self.enable_shear = enable_shear
        self.enable_hflip = enable_hflip
        self.enable_vflip = enable_vflip
        self.enable_zoom = enable_zoom
        self.enable_rotation = enable_rotation
        self.enable_resize = enable_resize
        
        self.transform_resize = Resize((img_resize[0],img_resize[1]), antialias='True')
        
    def horizontal_flip(self, image, bboxes):        
        #random horizontal flip
        trans = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
        image, bboxes, labels = trans(image, bboxes, np.ones((bboxes.shape[0], 1)))
        return image, bboxes
    
    def vertical_flip(self, image, bboxes):        
        #random horizontal flip
        trans = transforms.Compose([transforms.RandomVerticalFlip(p=0.5)])
        image, bboxes, labels = trans(image, bboxes, np.ones((bboxes.shape[0], 1)))
        return image, bboxes
    
    def affine_zoom(self, image, bboxes):        
        #random zoom
        if np.random.uniform(0, 1) > 0.5:
            image, bboxes = self.affine_transform(image, bboxes, scale = (self.zoom_range[0],self.zoom_range[1]))
        return image, bboxes
    
    def affine_shear(self, image, bboxes):        
        #random shear
        if np.random.uniform(0, 1) > 0.5:
            s_a1, s_a2 = self.shear_angle
            image, bboxes = self.affine_transform(image, bboxes, shear=(s_a1, s_a2,s_a1, s_a2))
        return image, bboxes
    
    def affine_rotation(self, image, bboxes):        
        #random shear
        if np.random.uniform(0, 1) > 0.5:
            image, bboxes = self.affine_transform(image, bboxes, degrees=(self.enable_rotation[0],self.enable_rotation[1]))
        return image, bboxes
    
    def affine_transform(self, image, bboxes, scale=(1,1), angle=0, translate=[0, 0], shear=0):
        trans = transforms.Compose([transforms.RandomAffine(scale=scale, degrees=angle, translate=translate, shear=shear)])
        #image = affine(image, scale=scale, angle=angle, translate=translate, shear=shear)
        image, bboxes, labels = trans(image, bboxes, np.ones((bboxes.shape[0], 1)))
        return image, bboxes            
    

        
    def run(self, image, points):

        bboxes = points_to_bboxes(points, image)
        if self.enable_resize:
            image = self.transform_resize(image)
        
        if self.enable_hflip:
            image, bboxes = self.horizontal_flip(image, bboxes)
            
        if self.enable_vflip:
            image, bboxes = self.vertical_flip(image, bboxes)
            
        if self.enable_zoom:
            image, bboxes = self.affine_zoom(image, bboxes)
            
        if self.enable_shear:
            image, bboxes = self.affine_shear(image, bboxes)

        if self.enable_rotation:
            image, bboxes = self.affine_transform(image, bboxes)

        points = bboxes[:,0:2].float()

        return image, points
    
def points_to_bboxes(points, img):
    ones_columns = np.ones((points.shape[0], 2))
    bboxes = np.hstack((points, ones_columns))
    bboxes = datapoints.BoundingBox(bboxes,
                                    format=datapoints.BoundingBoxFormat.XYWH,
                                    spatial_size=transforms.functional.get_spatial_size(img),
                                    )
    return bboxes
