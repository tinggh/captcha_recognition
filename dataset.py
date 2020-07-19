import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class folderDataset(Dataset):
    def __init__(self, img_dir, label_path, alphabet, resize, transforms=None):
        super(folderDataset, self).__init__()
        self.img_dir = img_dir
        self.labels = self.get_labels(label_path)
        self.alphabet = alphabet
        self.transforms = transforms
        self.width, self.height, self.channel = resize
    
    def get_labels(self, label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            labels = [{'name':c.strip('\n').split(' ')[0], 'lable':c.strip('\n').split(' ')[1]} for c in f.readlines()]	
        return labels


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_name= self.labels[index]['name']
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        h, w, c = image.shape
        assert c == self.channel, 'channel import error'
        image = cv2.resize(image, (0,0), fx=self.width/w, fy=self.height/h, interpolation=cv2.INTER_CUBIC)
        image = (np.reshape(image, (self.height, self.width, self.channel))).transpose(2, 0, 1)
        image = image.astype(np.float32) / 255.
        image = torch.from_numpy(image).type(torch.FloatTensor)
        transformer = transforms.Normalize([0.906, 0.910, 0.907], [0.147, 0.130, 0.142])
        image = transformer(image)
        return image, index

		


if __name__ == '__main__':
    import string
    alphabet = string.digits + string.ascii_letters

    val_dataset = folderDataset("/home/yjwang/Workspaces/data/captcha", "/home/yjwang/Workspaces/data/captcha/valid.txt", alphabet, (128, 64, 3))
    dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
	
    for i_batch, (image, index) in enumerate(dataloader):
        print(image.shape)
