#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 11:05:22 2017

@author: harik
"""

#%%

# import modules 

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import tarfile
import h5py
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from PIL import Image

np.random.seed(133)

#%% 

# MNIST dataset

# SVHN dataset 
svhn_url = 'http://ufldl.stanford.edu/housenumbers/'

#%%

# download, extract and display sample images

svhn_data = './svhn_data/'
if not os.path.isdir(svhn_data):    
    print ('Creating dir:', svhn_data)
    os.mkdir(svhn_data)

# reused/modified from tensorflow 1_notmnist.ipynb

last_percent_reported = None
def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        

def maybe_download(url, filename, expected_bytes, force=False):
  """Create dir if not present"""
  """Download a file if not present, and make sure it's the right size."""
  
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url+filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

svhn_train_filename = maybe_download(svhn_url, 'train.tar.gz', 404141560)
svhn_test_filename = maybe_download(svhn_url, 'test.tar.gz', 276555967)
svhn_extra_filename = maybe_download(svhn_url, 'extra.tar.gz', 1955489752)


def maybe_extract(filename, force=False):
  global svhn_data
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(path=svhn_data)
    tar.close()
  return root
  
svhn_train_folder = 'svhn_data/' + maybe_extract(svhn_train_filename)
svhn_test_folder  = 'svhn_data/' + maybe_extract(svhn_test_filename)
svhn_extra_folder = 'svhn_data/' + maybe_extract(svhn_extra_filename)

def display_samples(data_folder, num_samples=1):
    for i in range(num_samples):
        im_name = random.choice(os.listdir(data_folder))
        im_file = data_folder + "/" + im_name
        #display(Image(filename=im_file))

#display_samples(svhn_train_folder)
#display_samples(svhn_test_folder)
#display_samples(svhn_extra_folder)
 
#%%

# resued/modified: from https://github.com/ritchieng/NumNum/NumNum/load_data.py

# Create dictionary for bounding boxes
print('Creating dictionary of bounding boxes...')
class DigitStructFile:
    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']

    def getName(self,n):
        return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]].value])

    def bboxHelper(self,attr):
        if (len(attr) > 1):
            attr = [self.inf[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr

    def getBbox(self,n):
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['height'] = self.bboxHelper(self.inf[bb]["height"])
        bbox['label'] = self.bboxHelper(self.inf[bb]["label"])
        bbox['left'] = self.bboxHelper(self.inf[bb]["left"])
        bbox['top'] = self.bboxHelper(self.inf[bb]["top"])
        bbox['width'] = self.bboxHelper(self.inf[bb]["width"])
        return bbox
    
    def getDigitStructure(self,n):
        s = self.getBbox(n)
        s['name']=self.getName(n)
        return s

    def getAllDigitStructure(self):
        return [self.getDigitStructure(i) for i in range(len(self.digitStructName))]

    def getAllDigitStructure_ByDigit(self):
        pictDat = self.getAllDigitStructure()
        result = []
        structCnt = 1
        for i in range(len(pictDat)):
            item = { 'filename' : pictDat[i]["name"] }
            figures = []
            for j in range(len(pictDat[i]['height'])):
               figure = {}
               figure['height'] = pictDat[i]['height'][j]
               figure['label']  = pictDat[i]['label'][j]
               figure['left']   = pictDat[i]['left'][j]
               figure['top']    = pictDat[i]['top'][j]
               figure['width']  = pictDat[i]['width'][j]
               figures.append(figure)
            structCnt = structCnt + 1
            item['boxes'] = figures
            result.append(item)
        return result
    
print("Successfully created dictionary of bounding boxes!")
   
# Get Digit Structure
print('Getting digit structure for training data...')
digitFileTrain=DigitStructFile(os.path.join(svhn_train_folder,'digitStruct.mat'))
train_data=digitFileTrain.getAllDigitStructure_ByDigit()
print('Success!')

print('Getting digit structure for test data...')
digitFileTest=DigitStructFile(os.path.join(svhn_test_folder,'digitStruct.mat'))
test_data=digitFileTest.getAllDigitStructure_ByDigit()
print('Success!')

print('Getting digit structure for extra data...')
digitFileExtra=DigitStructFile(os.path.join(svhn_extra_folder,'digitStruct.mat'))
extra_data=digitFileExtra.getAllDigitStructure_ByDigit()
print('Success!')

# Crop Training Images
print('Cropping training images...')
train_imsize = np.ndarray([len(train_data),2])
for i in np.arange(len(train_data)):
    filename = train_data[i]['filename']
    fullname = os.path.join(svhn_train_folder, filename)
    im = Image.open(fullname)
    train_imsize[i, :] = im.size[:]

print('Success!')

# Crop Test Images
print('Cropping test images...')
test_imsize = np.ndarray([len(test_data),2])
for i in np.arange(len(test_data)):
    filename = test_data[i]['filename']
    fullname = os.path.join(svhn_test_folder, filename)
    im = Image.open(fullname)
    test_imsize[i, :] = im.size[:]

print('Success!')

# Crop Extra Images
print('Cropping extra images...')
extra_imsize = np.ndarray([len(extra_data),2])
for i in np.arange(len(extra_data)):
    filename = extra_data[i]['filename']
    fullname = os.path.join(svhn_extra_folder, filename)
    im = Image.open(fullname)
    extra_imsize[i, :] = im.size[:]

print('Success!')


def generate_dataset(data, folder):
    dataset = np.ndarray([len(data),32,32,1], dtype='float32')
    dataset_orig = np.ndarray([len(data),32,32,1], dtype='float32')
    labels = np.ones([len(data),6], dtype=int) * 10
    bboxes = np.zeros([len(data),4], dtype='float32')
    for i in np.arange(len(data)):
        filename = data[i]['filename']
        fullname = os.path.join(folder, filename)
        im = Image.open(fullname)
        boxes = data[i]['boxes']
        num_digit = len(boxes)
        labels[i,0] = num_digit
        top = np.ndarray([num_digit], dtype='float32')
        left = np.ndarray([num_digit], dtype='float32')
        height = np.ndarray([num_digit], dtype='float32')
        width = np.ndarray([num_digit], dtype='float32')
        for j in np.arange(num_digit):
            if j < 5: 
                labels[i,j+1] = boxes[j]['label']
                if boxes[j]['label'] == 10: labels[i,j+1] = 0
            else: print('#',i,'image has more than 5 digits.')
            top[j] = boxes[j]['top']
            left[j] = boxes[j]['left']
            height[j] = boxes[j]['height']
            width[j] = boxes[j]['width']
        
        im_top = np.amin(top)
        im_left = np.amin(left)
        im_height = np.amax(top) + height[np.argmax(top)] - im_top
        im_width = np.amax(left) + width[np.argmax(left)] - im_left
        
        im_top = int(np.floor(im_top - 0.1 * im_height))
        im_left = int(np.floor(im_left - 0.1 * im_width))
        im_bottom = int(np.amin([np.ceil(im_top + 1.2 * im_height), im.size[1]]))
        im_right = int(np.amin([np.ceil(im_left + 1.2 * im_width), im.size[0]]))
        bboxes[i,:] = [im_top, im_left, im_bottom, im_right]

        im_orig = im
        im_orig = im_orig.resize((32, 32), Image.ANTIALIAS)        
        mean = np.mean(im_orig, dtype='float32')
        std = np.std(im_orig, dtype='float32', ddof=1)
        if std < 1e-4: std = 1.
        im_orig = (im_orig - mean) / std
        dataset_orig[i,:,:,:] = im_orig[:,:,:]

        im = im.crop((im_left, im_top, im_right, im_bottom)).resize([32,32], Image.ANTIALIAS)
        im = np.dot(np.array(im, dtype='float32'), [[0.2989],[0.5870],[0.1140]])
        mean = np.mean(im, dtype='float32')
        std = np.std(im, dtype='float32', ddof=1)
        if std < 1e-4: std = 1.
        im = (im - mean) / std
        dataset[i,:,:,:] = im[:,:,:]

    return dataset, dataset_orig, labels, bboxes

print('Generating training dataset and labels...')
train_dataset, train_dataset_orig, train_labels, train_bboxes = generate_dataset(train_data, svhn_train_folder)
print('Success! \n Training set: {} \n Training labels: {}'.format(train_dataset.shape, train_dataset_orig.shape, train_labels.shape, train_bboxes.shape))

print('Generating testing dataset and labels...')
test_dataset, test_dataset_orig, test_labels, test_bboxes = generate_dataset(test_data, svhn_test_folder)
print('Success! \n Testing set: {} \n Testing labels: {}'.format(test_dataset.shape, test_dataset_orig.shape, test_labels.shape, test_bboxes.shape))

print('Generating extra dataset and labels...')
extra_dataset, extra_dataset_orig, extra_labels, extra_bboxes = generate_dataset(extra_data, svhn_extra_folder)
print('Success! \n Testing set: {} \n Testing labels: {}'.format(extra_dataset.shape, extra_dataset_orig.shape, extra_labels.shape, extra_bboxes.shape))

# Clean up data by deleting digits more than 5 (very few)
print('Cleaning up training data...')
train_dataset = np.delete(train_dataset, 29929, axis=0)
train_dataset_orig = np.delete(train_dataset_orig, 29929, axis=0)
train_labels = np.delete(train_labels, 29929, axis=0)
train_bboxes = np.delete(train_bboxes, 29929, axis=0)
print('Success!')

#%%
# Expand Training Data
print('Expanding training data randomly...')

random.seed(8)

n_labels = 10
valid_index = []
valid_index2 = []
train_index = []
train_index2 = []
for i in np.arange(n_labels):
    valid_index.extend(np.where(train_labels[:,1] == (i))[0][:400].tolist())
    train_index.extend(np.where(train_labels[:,1] == (i))[0][400:].tolist())
    valid_index2.extend(np.where(extra_labels[:,1] == (i))[0][:200].tolist())
    train_index2.extend(np.where(extra_labels[:,1] == (i))[0][200:].tolist())

random.shuffle(valid_index)
random.shuffle(train_index)
random.shuffle(valid_index2)
random.shuffle(train_index2)

valid_dataset = np.concatenate((extra_dataset[valid_index2,:,:,:], train_dataset[valid_index,:,:,:]), axis=0)
valid_dataset_orig = np.concatenate((extra_dataset_orig[valid_index2,:,:,:], train_dataset_orig[valid_index,:,:,:]), axis=0)
valid_labels = np.concatenate((extra_labels[valid_index2,:], train_labels[valid_index,:]), axis=0)
valid_bboxes = np.concatenate((extra_bboxes[valid_index2,:], train_bboxes[valid_index,:]), axis=0)

train_dataset_new = np.concatenate((extra_dataset[train_index2,:,:,:], train_dataset[train_index,:,:,:]), axis=0)
train_dataset_orig_new = np.concatenate((extra_dataset_orig[train_index2,:,:,:], train_dataset_orig[train_index,:,:,:]), axis=0)
train_labels_new = np.concatenate((extra_labels[train_index2,:], train_labels[train_index,:]), axis=0)
train_bboxes_new = np.concatenate((extra_bboxes[train_index2,:], train_bboxes[train_index,:]), axis=0)

print('Success! \n Training set: {} \n Training labels: {}'.format(train_dataset_new.shape, train_dataset_orig_new.shape, train_labels_new.shape, train_bboxes_new.shape))
print('Success! \n Validation set: {} \n Validation labels: {}'.format(valid_dataset.shape, valid_labels.shape, valid_bboxes.shape))
print('Success! \n Testing set: {} \n Testing labels: {}'.format(test_dataset.shape, test_labels.shape, test_bboxes.shape))

#%%

# Create Pickling File
print('Pickling data...')
pickle_file = 'svhn_data/SVHN.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_dataset_orig': train_dataset_orig,
        'train_labels': train_labels,
        'train_bboxes': train_bboxes,
        'valid_dataset': valid_dataset,
        'valid_dataset_orig': valid_dataset_orig,
        'valid_labels': valid_labels,
        'valid_bboxes': valid_bboxes,
        'test_dataset': test_dataset,
        'test_dataset_orig': test_dataset_orig,
        'test_labels': test_labels,
        'test_bboxes': test_bboxes,
        }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to {}: {}'.format(pickle_file, e))
    raise
    
statinfo = os.stat(pickle_file)
print('Success!')
print('Compressed pickle size: {}'.format(statinfo.st_size))

