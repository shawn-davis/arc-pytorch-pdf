"""
taken and modified from https://github.com/pranv/ARC
"""

import os
import numpy as np
from numpy.random import choice
import torch
from torch.autograd import Variable
import torchvision
from torchvision import transforms   

from scipy.misc import imresize as resize

from image_augmenter import ImageAugmenter

#use_cuda = False
device = torch.device("cpu")

class Morrowind(object):
    def __init__(self, batch_size=128):
        t = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        deManual = torch.stack([x for (x, y) in torchvision.datasets.ImageFolder(root='./rendered/de/', transform=t)], dim=1)[0]
        frManual = torch.stack([x for (x, y) in torchvision.datasets.ImageFolder(root='./rendered/fr/', transform=t)], dim=1)[0]
        engManual = torch.stack([x for (x, y) in torchvision.datasets.ImageFolder(root='./rendered/eng/', transform=t)], dim=1)[0]
        esManual = torch.stack([x for (x, y) in torchvision.datasets.ImageFolder(root='./rendered/es/', transform=t)], dim=1)[0]
        itManual = torch.stack([x for (x, y) in torchvision.datasets.ImageFolder(root='./rendered/it/', transform=t )], dim=1)[0]
        ukManual = torch.stack([x for (x, y) in torchvision.datasets.ImageFolder(root='./rendered/uk/', transform=t)], dim=1)[0]
        
        orig_chars = np.array(torch.stack([deManual, frManual, engManual, esManual, itManual, ukManual], dim=1).numpy() * 255, dtype='uint8')
        
        chars = np.zeros((54, 6, 1418, 906), dtype='uint8')
        for i in range(27):
            for j in range(6):
                chars[i * 2, j] = orig_chars[i, j][0:1418, 0:906]
                chars[i * 2 + 1, j] = orig_chars[i, j][0:1418, 906:1812]      
    
        self.mean_pixel = chars.mean() / 255.0
        
        self.data = chars
        self.height = chars.shape[2]
        self.width = chars.shape[3]
        self.batch_size = batch_size
        
        flip = False
        scale = 0.05
        rotation_deg = 0
        shear_deg = 0
        translation_px = 5
        
        self.augmentor = ImageAugmenter(self.width, self.height,
                            hflip=flip, vflip=flip,
                            scale_to_percent=1.0 + scale, rotation_deg=rotation_deg, shear_deg=shear_deg,
                            translation_x_px=translation_px, translation_y_px=translation_px)

    def fetch_batch(self, part):
        """
            This outputs batch_size number of pairs
            Thus the actual number of images outputted is 2 * batch_size
            Say A & B form the half of a pair
            The Batch is divided into 4 parts:
                Dissimilar A 		Dissimilar B
                Similar A 			Similar B

            Corresponding images in Similar A and Similar B form the similar pair
            similarly, Dissimilar A and Dissimilar B form the dissimilar pair

            When flattened, the batch has 4 parts with indices:
                Dissimilar A 		0 - batch_size / 2
                Similar A    		batch_size / 2  - batch_size
                Dissimilar B 		batch_size  - 3 * batch_size / 2
                Similar B 			3 * batch_size / 2 - batch_size

        """
        pass


class Batcher(Morrowind):
    def __init__(self, batch_size=128):
        Morrowind.__init__(self, batch_size)

    def fetch_batch(self, part, batch_size: int = None):

        if batch_size is None:
            batch_size = self.batch_size

        X, Y = self._fetch_batch(part, batch_size)

        X = Variable(torch.from_numpy(X)).view(2*batch_size, self.height, self.width)

        X1 = X[:batch_size]  # (B, h, w)
        X2 = X[batch_size:]  # (B, h, w)

        X = torch.stack([X1, X2], dim=1)  # (B, 2, h, w)

        Y = Variable(torch.from_numpy(Y))

        X, Y = torch.tensor(X, device=device), torch.tensor(Y, device=device)
#        if use_cuda:
#            X, Y = X.cuda(), Y.cuda()

        return X, Y

    def _fetch_batch(self, part, batch_size: int = None):
        if batch_size is None:
            batch_size = self.batch_size

        data = self.data
        height = self.height
        width = self.width

        X = np.zeros((2 * batch_size, height, width), dtype='uint8')
        for i in range(batch_size // 2):
            # choose similar pages
            same_page_num = choice(54)
            lang_for_sim = choice(6)

            # choose dissimilar pages
            diff_page_nums = choice(54, 2, replace=False)
            langs_for_dissim = choice(6, 2, replace=True)

            X[i], X[i + batch_size] = data[diff_page_nums, langs_for_dissim]
            X[i + batch_size // 2], X[i + 3 * batch_size // 2] = data[same_page_num, [lang_for_sim, lang_for_sim]]

        y = np.zeros((batch_size, 1), dtype='int32')
        y[:batch_size // 2] = 0
        y[batch_size // 2:] = 1

        if part == 'train':
            X = self.augmentor.augment_batch(X)
        else:
            X = X / 255.0

        X = X - self.mean_pixel
        X = X[:, np.newaxis]
        X = X.astype("float32")

        return X, y

