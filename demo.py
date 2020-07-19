import argparse
import os
import string
import sys
import time

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import utils
import crnn_captcha

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./crnn_capcha.pth', help='the path to your images')
parser.add_argument('--imgs_dir', type=str, default='./imgs', help='the path to your images')

opt = parser.parse_args()



crnn_model_path = opt.model_path
alphabet = string.digits + string.ascii_letters
nclass = len(alphabet)+1

transformer = transforms.Normalize([0.906, 0.910, 0.907], [0.147, 0.130, 0.142])
imgH = 64

def crnn_recognition(image, model):

    converter = utils.strLabelConverter(alphabet)
    h, w, c = image.shape
    ratio = imgH * 1.0/h
    image = cv2.resize(image, (0,0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    image = (np.reshape(image, (imgH, -1, c))).transpose(2, 0, 1)
    image = image.astype(np.float32) / 255.
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

   
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred


if __name__ == '__main__':
    model = crnn_captcha.CRNN(64, 3, nclass, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from {0}'.format(crnn_model_path))
    model.load_state_dict(torch.load(crnn_model_path))
    model.eval()
    started = time.time()
    imgs = os.listdir(opt.imgs_dir)
    n = 0
    for i, name in enumerate(imgs):
        img_path = os.path.join(opt.imgs_dir, name)
        image = cv2.imread(img_path)
        pred = crnn_recognition(image, model)
        print('path: {0}, pred: {1}'.format(name, pred))
    finished = time.time()
    print('elapsed time: {0}'.format(finished-started))
