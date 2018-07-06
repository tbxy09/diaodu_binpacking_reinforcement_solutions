import numpy as np
from torch.utils import data

class Dst(data.Dataset):
    def __init__(self,s,label):
        self.s=s
        self.label=label
    def __getitem__(self,index):
        import torch
        tmp=self.s[index][:,-1]
        tmp=(tmp-tmp.mean())/tmp.std()
        s_norm=np.hstack([self.s[index][:,:-1],tmp.reshape(-1,1)])
#         return self.s[index],self.label[index]
#         return tmp.reshape(-1,1),self.label[index]
        return s_norm,self.label[index]

    def __len__(self):
        return len(self.label)
