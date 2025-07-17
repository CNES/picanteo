from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import rasterio
from rasterio.windows import Window
import numpy as np
import torch 
from picanteo.utils.logger import logger

class InferenceDataset(Dataset):
    def __init__(
        self,
        img_path,
        patch_size,
        overlap,
        padding,
        shifted_border
    ) :
        """Initialize a new dataset instance.

        Args:
   

        """
        self.img_path = img_path
        self.patch_size = patch_size
        self.overlap = overlap
        self.padding = padding
        self.shifted_border = shifted_border
        self.setup()
        
    def setup(self):
        pass
    
    def crop(self):
        pass
    
    def __getitem__(self, index: int):
        pass
    
    def __len__(self):
        pass
    
class FullInferenceDataset(InferenceDataset):
    def __init__(
        self,
        img_path,
        patch_size,
        overlap,
        padding,
        shifted_border
    ) :
        """Initialize a new  dataset instance.

        Args:
        """
        super().__init__(img_path,patch_size,overlap,padding,shifted_border)
        
        self.dataset = self.setup()
        logger.debug(self.dataset)

    def setup(self):
        requires_padding = False
        with rasterio.open(self.img_path, mode='r') as src:
           
            self.crop_width = src.width
            self.crop_height = src.height
        
        
        dataset=[(self.img_path, 0, 0,src.width, src.height, requires_padding)]
        return dataset
    
    def pad_image(self, image):
        
        C, H, W = image.shape

        H_target = ((H + 31) // 32) * 32
        W_target = ((W + 31) // 32) * 32
        target = max(H_target, W_target)
        pad_h = H_target - H
        pad_w = W_target - W

        padded_image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='symmetric')

        return padded_image




    def unpad_image(self, padded_image):
        H_original, W_original = img.shape[0], img.shape[1]

        unpadded_image = padded_image[:H_original, :W_original, :]

        return unpadded_image
    
    def __getitem__(self, index: int):
        """Get item.

        Args:
            index (int): Index of the element to retrieve.

        Returns:
            sample: patch with windowed row,col information

        """
        
      
        img_name, col, row, patch_size_col, patch_size_row, padded_patch = self.dataset[index]
  
        with rasterio.open(self.img_path, mode='r') as src:

          
            img = src.read(window=Window(col, row, patch_size_col,  patch_size_row))
           
            img = self.pad_image(img)
            
            img = img.astype(np.float32)
            
            image = torch.from_numpy(img)


        sample = {
            "image": image,
            "col": col,
            "row": row,
            "patch_size_col": patch_size_col,
            "patch_size_row": patch_size_row,
            "padded_patch": padded_patch
        }
        return sample
    
    def __len__(self) :
        """Return the length of the dataset."""
        return len(self.dataset)
    
class TiledInferenceDataset(InferenceDataset):

    def __init__(
        self,
        img_path,
        patch_size,
        overlap,
        padding,
        shifted_border
    ) :
        """Initialize a new dataset instance.

        Args:

        """
        super().__init__(img_path,patch_size,overlap,padding,shifted_border)
     
        
    def setup(self):
        
        with rasterio.open(self.img_path, mode='r') as src:
          
            self.crop_width = self.patch_size
            self.crop_height = self.patch_size
            
        if self.shifted_border:
            self.dataset = self.get_tiled_dataset_full() 
        else:
            self.dataset = self.get_tiled_dataset_crop()

        logger.info("len dataset: " + str(len(self.dataset)))
        logger.debug(self.dataset)
  

    def __getitem__(self, index: int):
        """Get item.

        Args:
            index (int): Index of the element to retrieve.

        Returns:
            sample: patch with windowed row,col information

        """
     
        img_name, col, row, patch_size_col, patch_size_row, padded_patch = self.dataset[index]
 
        with rasterio.open(self.img_path, mode='r') as src:
            
            img = src.read(window=Window(col, row, self.patch_size, self.patch_size))
 
            img = img.astype(np.float32)
            image = torch.from_numpy(img)
        

        sample = {
            "image": image,
            "col": col,
            "row": row,
            "patch_size_col": patch_size_col,
            "patch_size_row": patch_size_row,
            "padded_patch": padded_patch
        }
        return sample
    

    def get_tiled_dataset_full(self):
        with rasterio.open(self.img_path, mode='r') as src:
            image_width = src.width
            image_height = src.height
        step = self.patch_size-self.overlap

        dataset = []

        end_x = image_width +1 - self.patch_size
        indices_x = list(range(0, end_x, step))
    
        if indices_x != image_width - self.patch_size:     
            indices_x.append(image_width - self.patch_size)
           
        nr_tiles_x = len(indices_x)
  

        end_y = image_height +1 - self.patch_size
        indices_y = list(range(0, end_y, step))
    
        
        if indices_y != image_height - self.patch_size:     
            indices_y.append(image_height - self.patch_size)
           
        nr_tiles_y= len(indices_y)
        



        offset_x = 0
        offset_y = 0

    

        logger.info(f"looping through {nr_tiles_y}, {nr_tiles_x} patches")
        dataset = []
        for ty in range(nr_tiles_y):
            for tx in range(nr_tiles_x):
                # Determine the window to extract
                start_x = int(offset_x + indices_x[tx])
                start_y = int(offset_y + indices_y[ty])
                dataset.append((self.img_path, start_x, start_y, self.patch_size, self.patch_size, False))

        return dataset
    
    def get_tiled_dataset_crop(self):
        with rasterio.open(self.img_path, mode='r') as src:
        
            image_width = src.width
            image_height = src.height
        step = self.patch_size-self.overlap

        dataset = []

        end_x = image_width +1 - self.patch_size
        indices_x = list(range(0, end_x, step))
       
        nr_tiles_x = len(indices_x)

        remaining_x =image_width - (indices_x[-1]+self.patch_size)


        end_y = image_height +1 - self.patch_size
        indices_y = list(range(0, end_y, step))
     
        nr_tiles_y = len(indices_y)
        remaining_y =image_height - (indices_y[-1]+self.patch_size)



        offset_x = int(remaining_x / 2)
        offset_y = int(remaining_y / 2)

    

        logger.info(f"looping through {nr_tiles_y}, {nr_tiles_x} patches")
        dataset = []
        for ty in range(nr_tiles_y):
            for tx in range(nr_tiles_x):
                # Determine the window to extract
                start_x = int(offset_x + indices_x[tx])
                start_y = int(offset_y + indices_y[ty])
                dataset.append((self.img_path, start_x, start_y,self.patch_size, self.patch_size, False))
           
        return dataset
    
    def __len__(self) :
        """Return the length of the dataset."""
        return len(self.dataset)