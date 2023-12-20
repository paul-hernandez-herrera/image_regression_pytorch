from torchvision.transforms.functional import affine, hflip, vflip
from torchvision.transforms import Resize
import torchvision.transforms.v2 as transforms
from torchvision import datapoints
import torchvision
import numpy as np
from torch import tensor

class augmentation_task():
    def __init__(self, zoom_range = [0.8, 1.2],
                 shear_angle = [-5, 5], 
                 img_resize = [224,224],
                 enable_shear = True, 
                 enable_hflip = True, 
                 enable_vflip = True, 
                 enable_zoom = True,
                 enable_resize = False):
        self.zoom_range = zoom_range
        self.shear_angle = shear_angle
        
        #flag to compute specific transformations
        self.enable_shear = enable_shear
        self.enable_hflip = enable_hflip
        self.enable_vflip = enable_vflip
        self.enable_zoom = enable_zoom
        self.enable_resize = enable_resize
        
        self.transform_resize = Resize((img_resize[0],img_resize[1]), antialias='True')
        
    def horizontal_flip(self, image, bboxes):        
        #random horizontal flip
        trans = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
        image, bboxes, labels = trans(image, bboxes, np.ones((bboxes.shape[0], 1)))
        return image, bboxes
    
    def vertical_flip(self, image):        
        #random horizontal flip
        print("Doing vertical flip")
        trans = transforms.Compose([transforms.RandomVerticalFlip(p=0)])
        image, bboxes, labels = trans(image, bboxes, np.ones((bboxes.shape[0], 1)))
        return image, bboxes
    
    def affine_transform(self, image, scale=1, angle=0, translate=[0, 0], shear=0):
        image = affine(image, scale=scale, angle=angle, translate=translate, shear=shear)
        image = affine(image, scale=scale, angle=angle, translate=translate, shear=shear)
        return image    
    
    def affine_zoom(self, image):        
        #random zoom
        if np.random.uniform(0, 1) > 0.5:
            zoom = np.random.uniform(*self.zoom_range)
            image = self.affine_transform(image, scale=zoom)
        return image
    
    def affine_shear(self, image):        
        #random shear
        if np.random.uniform(0, 1) > 0.5:
            shear = np.random.uniform(*self.shear_angle)
            image = self.affine_transform(image, shear=shear)
        return image
        
    def run(self, image, points):

        bboxes = points_to_bboxes(points, image)
        if self.enable_resize:
            image = self.transform_resize(image)
        
        if self.enable_hflip:
            image, bboxes = self.horizontal_flip(image, bboxes.numpy())
            
        if self.enable_vflip:
            image = self.vertical_flip(image)
            
        if self.enable_zoom:
            print("ZOOM")
            image = self.affine_zoom(image)
            
        if self.enable_shear:
            image = self.affine_shear(image)
        
        points = tensor(bboxes[:,0:2]).float()

        print(f"type points: {type(points)}" )
        return image, points
    
def points_to_bboxes(points, img):
    ones_columns = np.ones((points.shape[0], 2))
    bboxes = np.hstack((points, ones_columns))
    bboxes = datapoints.BoundingBox(bboxes,
                                    format=datapoints.BoundingBoxFormat.XYXY,
                                    spatial_size=transforms.functional.get_spatial_size(img),
                                    )
    return bboxes
