from torch.utils.data import Dataset
import os
import torch
import scipy.io as scio
import numpy as np


class Imgdataset(Dataset):

    def __init__(self, path):
        super(Imgdataset, self).__init__()
        self.data = []
        
        if os.path.exists(path):
            gt_path = path

            if os.path.exists(gt_path):
                gt = os.listdir(gt_path)
                self.data = [{'orig': gt_path + '/' + gt[i]} for i in range(len(gt))]
            else:
                raise FileNotFoundError('path doesnt exist!')
        else:
            raise FileNotFoundError('path doesnt exist!')

    def __getitem__(self, index,):
         data = self.data[index]["orig"]
         if data[-3:] == 'mat':
            data = scio.loadmat(data)
            
            
            otfr = np.expand_dims(data['otf_real'], axis=0)
            otfim = np.expand_dims(data['otf_imag'], axis=0)
            img = np.concatenate((otfr,otfim),axis=0)
            img = np.concatenate((img,otfim*0),axis=0)
            #padding image from input to 512x512
            img = np.pad(img,((0,0),(0,512-img.shape[-1]),(0,512-img.shape[-1])),mode='constant',constant_values=0)
            resol = img.shape[-1]
            img = img*resol**2
            img = torch.from_numpy(img).float()

            #slp = torch.from_numpy(data['slopes'])
            slpm = data['slp_mods']
            mean_slpm = np.expand_dims(np.mean(slpm,1),axis=1)
            std_slpm = np.expand_dims(np.std(slpm,1),axis=1)
            z = np.concatenate((np.expand_dims(slpm[:,0],axis=1),mean_slpm),axis=0)
            z = np.concatenate((z,std_slpm),axis=0).squeeze()*1e9
            z = np.concatenate((slpm.shape,z),axis=0)
            z = torch.from_numpy(z).float()
            
         elif data[-3:] == 'npy':
            data = np.load(data, allow_pickle=True)
            meas = data[0]
            gt = data[1]
         

         return img, z

    def __len__(self):

        return len(self.data)
