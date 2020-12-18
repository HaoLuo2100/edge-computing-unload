# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 20:51:32 2020

@author: John
作用：筛选训练集每个类别子文件夹，抽取500张放到新文件夹；筛选测试集每个类别子文件夹，抽取100张放到新文件夹
"""
import os
import sys
import shutil
import random


def copyimage(src, dst, param):
    # param == 'train' or 'valid'
    if not os.path.exists(os.path.join(dst, param)):
        os.makedirs(os.path.join(dst, param))
    if param != 'test':
        for root, dirs, files in os.walk(os.path.join(src, 'train')):
            if len(dirs) != 0:
                continue
            fileset = random.sample(files, 900)

            for idx, f in enumerate(fileset):
                sss = os.path.join(src, 'train', os.path.basename(root), f)
                # ddd = os.path.join(dst, 'train', os.path.basename(root))
                # if not os.path.exists(ddd):
                #     os.makedirs(ddd)
                # select 80 images from each train subset as training set
                if idx < 800:
                    ddd = os.path.join(dst, 'train', os.path.basename(root))
                    if not os.path.exists(ddd):
                        os.makedirs(ddd)
                # select 10 images from each train subset as valid set
                else:
                    ddd = os.path.join(dst, 'valid', os.path.basename(root))
                    if not os.path.exists(ddd):
                        os.makedirs(ddd)
                try:
                    shutil.copy(sss, ddd)
                except IOError as e:
                    print("Unable to copy file. %s" % e)
                except:
                    print("Unexpected error:", sys.exc_info())

    else:
        for root, dirs, files in os.walk(os.path.join(src, 'valid')):
            if len(dirs) != 0:
                continue
            ddd = os.path.join(dst, 'test', os.path.basename(root))
            if not os.path.exists(ddd):
                os.makedirs(ddd)
            # select 10 images from each valid subset as test set
            filesubset = random.sample(files, 100)
            for f in filesubset:
                sss = os.path.join(src, 'valid', os.path.basename(root), f)
                try:
                    shutil.copy(sss, ddd)
                except IOError as e:
                    print("Unable to copy file. %s" % e)
                except:
                    print("Unexpected error:", sys.exc_info())


if __name__ == '__main__':
    src = 'cifar-10'
    dst = 'cifar-10-subset'
    copyimage(src, dst, 'train')
    # copyimage(src, dst, 'valid')
    copyimage(src, dst, 'test')