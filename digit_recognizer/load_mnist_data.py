#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
#%%
"""

# import modules

import numpy as np
import os
import random
import sys
import gzip
import idx2numpy 
import matplotlib.pyplot as plt
from IPython.display import display, Image
from scipy import ndimage
from scipy.misc import imsave
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from socket import socket

# global variables

idisplay = 1
num_merge = 5
img_height = 28
img_width = 28

np.random.seed(133)

# MNIST dataset
mnist_url = 'http://yann.lecun.com/exdb/mnist/'

#%%

# download, extract and display sample images

mnist_data = './mnist_data/'
if not os.path.isdir(mnist_data):    
    print ('Creating dir:', mnist_data)
    os.mkdir(mnist_data)

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
  """Download a file if not present, and make sure it's the right size."""
  
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url+filename, mnist_data+filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return mnist_data+filename

mnist_train_images_gz = maybe_download(mnist_url, 'train-images-idx3-ubyte.gz', 9912422)
mnist_train_labels_gz = maybe_download(mnist_url, 'train-labels-idx1-ubyte.gz', 28881)
mnist_test_images_gz  = maybe_download(mnist_url, 't10k-images-idx3-ubyte.gz', 1648877)
mnist_test_labels_gz  = maybe_download(mnist_url, 't10k-labels-idx1-ubyte.gz', 4542)

def maybe_extract(filename):
    fname = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isfile(fname):
        print('Skipping extraction of %s.' % (filename))
    else:
        print('Extracting %s ...' % (filename))
        cmd = 'gunzip {}'.format(filename)
        os.system(cmd)
    return fname

maybe_extract('mnist_data/train-images-idx3-ubyte.gz')
maybe_extract('mnist_data/train-labels-idx1-ubyte.gz')
maybe_extract('mnist_data/t10k-images-idx3-ubyte.gz')
maybe_extract('mnist_data/t10k-labels-idx1-ubyte.gz')

train_samples = idx2numpy.convert_from_file('mnist_data/train-images-idx3-ubyte')
train_labels  = idx2numpy.convert_from_file('mnist_data/train-labels-idx1-ubyte')
test_samples  = idx2numpy.convert_from_file('mnist_data/t10k-images-idx3-ubyte')
test_labels   = idx2numpy.convert_from_file('mnist_data/t10k-labels-idx1-ubyte')

def display_samples(data, labels, text=None, num_samples=1, idx='rand', squeeze=0):
    for i in range(num_samples):
        if idx == 'rand':
            idx = random.choice(range(data.shape[0]))

        print 'Display sample {} image: index: {} label: {}'.format(text, idx, labels[idx])
        if squeeze == 0:
            plt.imshow(data[idx], interpolation='nearest')
        else:
            plt.imshow(np.squeeze(data[idx], axis=(2,)), interpolation='nearest')
        
        plt.show()
        
if idisplay:
    display_samples(train_samples, train_labels, text='train', num_samples=1)
    display_samples(test_samples, test_labels, text='test', num_samples=1)
    
def createSequences(data, labels, img_height, img_width, merge=5):
    num_merged = int(data.shape[0]/merge)
    nlabels = np.ndarray(shape=(data.shape[0]*2, merge+1), dtype=np.int32)
    ndata = np.ndarray(shape=(data.shape[0]*2, img_height, img_width*merge), dtype=np.float32)
    
    for i in range(merge*2):
        rand_idx = random.sample(range(0, data.shape[0]), data.shape[0])
        w = 0; 
        for j in range(num_merged):
            a, b, c, d, e = rand_idx[w:w+merge]
            num_rand = random.choice(range(1,6))
            zeros = np.zeros([28,28])
            if num_rand == 5: 
                ndata[(i*num_merged)+j,:,:] = np.concatenate([data[a],data[b],data[c],data[d],data[e]], axis=1)
                nlabels[(i*num_merged)+j,:] = np.hstack([0,labels[a],labels[b],labels[c],labels[d],labels[e]])
            elif num_rand == 4:
                ndata[(i*num_merged)+j,:,:] = np.concatenate([data[a],data[b],data[c],data[d],zeros], axis=1)
                nlabels[(i*num_merged)+j,:] = np.hstack([0,labels[a],labels[b],labels[c],labels[d],10])
            elif num_rand == 3:
                ndata[(i*num_merged)+j,:,:] = np.concatenate([data[a],data[b],data[c],zeros,zeros], axis=1)
                nlabels[(i*num_merged)+j,:] = np.hstack([0,labels[a],labels[b],labels[c],10,10])
            elif num_rand == 2:
                ndata[(i*num_merged)+j,:,:] = np.concatenate([data[a],data[b],zeros,zeros,zeros], axis=1)
                nlabels[(i*num_merged)+j,:] = np.hstack([0,labels[a],labels[b],10,10,10])
            elif num_rand == 1:
                ndata[(i*num_merged)+j,:,:] = np.concatenate([data[a],zeros,zeros,zeros,zeros], axis=1)
                nlabels[(i*num_merged)+j,:] = np.hstack([0,labels[a],10,10,10,10])
            w += merge

    # add dim for grey scale
    ndata = np.expand_dims(ndata, axis=3)
    
    return ndata, nlabels
    
m_train_samples, m_train_labels = createSequences(train_samples, 
                                                  train_labels, 
                                                  img_height=img_height, 
                                                  img_width=img_width, 
                                                  merge=num_merge)

m_test_samples, m_test_labels = createSequences(test_samples, 
                                                test_labels, 
                                                img_height=img_height, 
                                                img_width=img_width, 
                                                merge=num_merge)

if idisplay:
    display_samples(m_train_samples, m_train_labels, text='train', num_samples=1, squeeze=1)
    display_samples(m_test_samples, m_test_labels, text='test', num_samples=1, squeeze=1)

mnist_merged = './mnist_merged/'
mnist_merged_train = './mnist_merged/merged_train'
mnist_merged_test  = './mnist_merged/merged_test'

if not os.path.isdir(mnist_merged):    
    print ('Creating dir:', mnist_merged)
    os.mkdir(mnist_merged)
    os.mkdir(mnist_merged_train)
    os.mkdir(mnist_merged_test)

print 'Saving merged train images to: {} ...'.format(mnist_merged_train)
for i in range(m_train_samples.shape[0]):
    save_file = '{}/{}.png'.format(mnist_merged_train, i)
    imsave(save_file, (np.squeeze(m_train_samples[i], axis=(2,))))

print 'Saving merged test images to: {} ...'.format(mnist_merged_test)
for i in range(m_test_samples.shape[0]):
    save_file = '{}/{}.png'.format(mnist_merged_test, i)
    imsave(save_file, (np.squeeze(m_test_samples[i], axis=(2,))))

if idisplay:
    Image(filename='mnist_merged/merged_train/1.png')
    Image(filename='mnist_merged/merged_test/1.png')

# Save
m_train_samples_save = m_train_samples
m_train_labels_save = m_train_labels

# Create validation set (6000 of 60000)
nval = 6000
m_val_samples = np.ndarray(shape=(nval, img_height, img_width*num_merge), dtype=np.float32)
m_val_labels = np.ndarray(shape=(nval, num_merge), dtype=np.int32)

m_val_samples = m_train_samples[:nval,]
m_val_labels = m_train_labels[:nval,]
m_train_samples = np.delete(m_train_samples, np.r_[:nval], 0)
m_train_labels = np.delete(m_train_labels, np.r_[:nval], 0)

# Create Pickling File
print('Pickling data...')
pickle_file = 'mnist_merged/MNIST.merged.pickle'

print 'Train samples: {}'.format(m_train_samples.shape)
print 'Train labels: {}'.format(m_train_labels.shape)
print 'Validation samples: {}'.format(m_val_samples.shape)
print 'Validation labels: {}'.format(m_val_labels.shape)
print 'Test samples: {}'.format(m_test_samples.shape)
print 'Test labels: {}'.format(m_test_labels.shape)

try:
    f = open(pickle_file, 'wb')
    save = {
        'm_train_samples': m_train_samples,
        'm_train_labels': m_train_labels,
        'm_val_samples': m_val_samples,
        'm_val_labels': m_val_labels,
        'm_test_samples': m_test_samples,
        'm_test_labels': m_test_labels,
        }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to {}: {}'.format(pickle_file, e))
    raise
    
statinfo = os.stat(pickle_file)
print('Success!')
print('Compressed pickle size: {}'.format(statinfo.st_size))
