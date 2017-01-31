
idisplay = 0
img_width = 32
img_height = 32

# import modules

import numpy as np
import os
import random
import sys
import gzip
import idx2numpy 
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import ndimage
from scipy.misc import imsave
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from socket import socket
from PIL import Image

# resize data to 32 x 32

orig_data = './my_data/orig/'

resize_data = './my_data/resize/'
if not os.path.isdir(resize_data):    
    print ('Creating dir:', resize_data)
    os.mkdir(resize_data)

for i in range(1, 6):
    rd_fname = orig_data + '{}.png'.format(i)
    wr_fname = resize_data + '{}.png'.format(i)
    img = Image.open(rd_fname)
    basewidth = 32
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((32, 32), Image.ANTIALIAS)
    img.save(wr_fname)

# load data

my_labels = np.ndarray(shape=(5, 6), dtype=np.int32)
my_data = np.ndarray(shape=(5, 32, 32), dtype=np.float32)

for i in range(5):
    fname = resize_data + '{}.png'.format(i+1)
    img = Image.open(fname).convert('L')
    my_data[i,:,:] = np.array(img)
    
my_data = np.expand_dims(my_data, axis=3)

my_labels[0,:] = [0,9,10,10,10,10]
my_labels[1,:] = [0,2,3,10,10,10]
my_labels[2,:] = [0,1,3,5,10,10]
my_labels[3,:] = [0,8,2,0,9,10]
my_labels[4,:] = [0,1,0,1,4,5]

def display_samples(data, labels, idx):
    print labels[idx]
    plt.imshow(np.squeeze(data[idx], axis=(2,)), interpolation='nearest')    
    plt.show()
        
if idisplay:
    display_samples(my_data, my_labels, 0)
    display_samples(my_data, my_labels, 1)
    display_samples(my_data, my_labels, 2)
    display_samples(my_data, my_labels, 3)
    display_samples(my_data, my_labels, 4)

# Create Pickling File
print('Pickling data...')
pickle_file = 'my_data/my_data.pickle'

print 'Test samples: {}'.format(my_data.shape)
print 'Test labels: {}'.format(my_labels.shape)

try:
    f = open(pickle_file, 'wb')
    save = {'my_data': my_data, 'my_labels': my_labels}
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to {}: {}'.format(pickle_file, e))
    raise
    
statinfo = os.stat(pickle_file)
print('Success!')
print('Compressed pickle size: {}'.format(statinfo.st_size))



