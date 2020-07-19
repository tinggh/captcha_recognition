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

import crnn_captcha

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./models/crnn_Rec_done_2_0.957867412140575.pth', help='the path to your images')
parser.add_argument('--test_path', type=str, default='/home/user/Workspaces/data/captcha/test_labels.txt', help='the path to your images')

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

	# crnn network
    model = crnn_captcha.CRNN(64, 3, nclass, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from {0}'.format(crnn_model_path))
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(crnn_model_path))
    model.eval()
    started = time.time()
    ## read an image
    with open(opt.test_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    testdir = os.path.dirname(opt.test_path)
    n = 0
    for i, l in enumerate(lines):
        name, gt = l.strip('\n').split(' ')
        img_path = os.path.join(testdir, name)
        image = cv2.imread(img_path)
        pred = crnn_recognition(image, model)
        if pred != gt:
            print('path: {0}, pred: {1}, gt:{2}'.format(name, pred, gt))
            n += 1
    finished = time.time()
    print('elapsed time: {0}'.format(finished-started))
    print('accuracy: {0}'.format(1-n*1.0/len(lines)))
