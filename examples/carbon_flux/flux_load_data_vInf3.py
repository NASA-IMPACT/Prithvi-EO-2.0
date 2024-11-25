import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import os

NO_DATA = -0.9999
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)

def load_raster(path,if_img,crop=None):
        
        with rasterio.open(path) as src:
            img = src.read()

            # load  selected 6 bands for Sentinnel 2 (S2)
            if if_img==1:
                bands=[0,1,2,3,4,5]
                img = img[bands,:,:]

            img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)# update our NO_DATA with -0.9999 -- chips are already scaled
            #print("img size",img.shape)
            
            if crop:
                img = img[:, -crop[0]:, -crop[1]:]
        #print('return from load ras')
        return img


def preprocess_image(image,means,stds):
        
        # normalize image
        means1 = means.reshape(-1,1,1)  # Mean across height and width, for each channel
        stds1 = stds.reshape(-1,1,1)    # Std deviation across height and width, for each channel
        normalized = image.copy()
        normalized = ((image - means1) / stds1)
        
        normalized = torch.from_numpy(normalized.reshape(1, normalized.shape[0], 1, *normalized.shape[-2:])).to(torch.float32)
        #print('return from norm')
        return normalized


#consistent for HLS -- modified to add merra, flux -- these have z-score processing as hls
class flux_dataset(Dataset):

    def __init__(self,path,means,stds, merras_data, merra_means, merra_stds, gpp_mean, gpp_std, target):
        self.data_dir=path
        self.means=means
        self.stds=stds
        self.merras=merras_data
        self.merra_means=merra_means
        self.merra_stds=merra_stds
        self.gpp_means=gpp_mean
        self.gpp_stds=gpp_std
        self.target=target
        

    def __len__(self):
        return len(self.data_dir)
    
    
    def __getitem__(self,idx):
        
        image_path=self.data_dir[idx]
        if_image = 1
        image=load_raster(image_path,if_image,crop=(50, 50))
        #print('hls after load raster', image.shape)
        final_image=preprocess_image(image,self.means,self.stds)
        #print('hls after preprocess', final_image.shape)
        merra=self.merras[idx]
        merra_vars = torch.from_numpy(np.array(self.merras[idx]).reshape(10, 1, 1)).to(torch.float32)
        
        mean_merra = self.merra_means.reshape(-1,1,1)  # Mean across height and width, for each channel
        stds_merra = self.merra_stds.reshape(-1,1,1)    # Std deviation across height and width, for each channel
        merra_vars_norm=(merra_vars-mean_merra)/(stds_merra)

        final_image=final_image.squeeze(0)
        
        mean_gpp = self.gpp_means.reshape(-1,1,1)  # Mean across height and width, for each channel
        stds_gpp = self.gpp_stds.reshape(-1,1,1)    # Std deviation across height and width, for each channel
        #print('mean, std gpp', mean_gpp, stds_gpp)
        gpp_vars_norm=(self.target[idx]-mean_gpp)/(stds_gpp)
        gpp_vars_norm=torch.from_numpy(np.array(gpp_vars_norm).reshape(1))
        #print('gpp is', gpp.shape)
        return final_image, merra_vars_norm, gpp_vars_norm



