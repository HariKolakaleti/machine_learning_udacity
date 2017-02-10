#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 22:19:29 2017

@author: harik
"""

""""
MNIST MODEL:

    -------------------------------------------------------------------------------
    CNN Model Architecture for multi digit recognition implemented with TensorFlow
    -------------------------------------------------------------------------------
      inputs    [batch_size, 28, 140, 1]
      conv1     [patch=3x15, stride=1x1, padding=valid, 16 features]
      relu1     [relu]
      maxpool1  [patch=2x2, stride=2x2, padding=valid]
      conv2     [patch=4x20, stride=1x1, padding=valid, 32 features]
      relu2     [relu]
      maxpool2  [patch=2x2, stride=2x2, padding=valid]
      conv3     [patch=5x22, stride=1x1, padding=valid, 96 features]
      relu2     [relu]
      drop_out  10 %
      fc        [nodes=64]
      outputs   [y1,y2,y3,y4,y5]

SVHN MODEL:

    -------------------------------------------------------------------------------
    CNN Model Architecture for multi digit recognition implemented with TensorFlow
    -------------------------------------------------------------------------------
      inputs    [batch_size, 32, 32, 1]
      conv1     [patch=5x5, stride=1x1, padding=valid, 16 features]
      relu1     [relu]
      maxpool1  [patch=2x2, stride=2x2, padding=valid]
      conv2     [patch=5x5, stride=1x1, padding=valid, 32 features]
      relu2     [relu]
      maxpool2  [patch=2x2, stride=2x2, padding=valid]
      conv3     [patch=5x5, stride=1x1, padding=valid, 96 features]
      relu2     [relu]
      drop_out  20 %
      fc        [nodes=64]
      outputs   [y1,y2,y3,y4,y5]

"""

#%%

# program mode control

debug     = 1
idisplay  = 0
svhn_en   = 0
mnist_en  = 0
mydata_en = 1
restore_session = 1

if mnist_en:
    num_steps  = 7125
    num_val    = 6000
    num_tests  = 10000
    img_width  = 140
    img_height = 28
    predict_bbox = 0
    session_name = 'session/digit_recognizer.ckpt'
elif svhn_en or mydata_en:
    num_steps  = 60000
    num_val    = 5684
    num_tests  = 13068
    img_width  = 32
    img_height = 32
    predict_bbox = 1
    localized_data = 1
    session_name = 'session/digit_recognizer.ckpt'

if mydata_en:
    restore_session = 1 

import pickle
import random
import numpy as np
import tensorflow as tf

if idisplay:
    import matplotlib.pyplot as plt
    from PIL import Image
    from IPython.display import display

if mnist_en:
    print 'Loading MNIST pickled data...'
    pickle_file = 'mnist_merged/MNIST.merged.pickle'
    
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        X_train_samples = save['m_train_samples']
        y_train_samples = save['m_train_labels']
        X_val_samples   = save['m_val_samples']
        y_val_samples   = save['m_val_labels']
        X_test_samples  = save['m_test_samples'][:4000,]
        y_test_samples  = save['m_test_labels'][:4000,]
        del save  
        print 'Training data shape: ', X_train_samples.shape
        print 'Training label shape:', y_train_samples.shape
        print 'Validation data shape:', X_val_samples.shape
        print 'Validation label shape:', y_val_samples.shape
        print 'Test data shape:     ', X_test_samples.shape
        print 'Test label shape:    ', y_test_samples.shape
        print 'Data successfully loaded !!'
elif svhn_en:
    print 'Loading SVHN pickled data...'
    pickle_file = 'svhn_data/SVHN.pickle'

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)

        if localized_data:            
            X_train_samples = save['train_dataset']
            X_val_samples   = save['valid_dataset']
            X_test_samples  = save['test_dataset']
        else:
            X_train_samples = save['train_dataset_orig']
            X_val_samples   = save['valid_dataset_orig']
            X_test_samples  = save['test_dataset_orig']
            
        if predict_bbox:
            y_train_samples = save['train_bboxes']
            y_val_samples = save['valid_bboxes']
            y_test_samples = save['test_bboxes']
        else:
            y_train_samples = save['train_labels']
            y_val_samples   = save['valid_labels']
            y_test_samples  = save['test_labels']        
        
        del save  
        print 'Training data shape: ', X_train_samples.shape
        print 'Training label shape:', y_train_samples.shape
        print 'Validation data shape:', X_val_samples.shape
        print 'Validation label shape:', y_val_samples.shape
        print 'Test data shape:     ', X_test_samples.shape
        print 'Test label shape:    ', y_test_samples.shape
        print 'Data successfully loaded !!'

elif mydata_en:
    print 'Loading mydata pickled data...'
    pickle_file = 'my_data/my_data.pickle'

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        # load same data to train/val for placeholders
        X_train_samples = save['my_data']
        y_train_samples = save['my_labels']
        X_val_samples   = save['my_data']
        y_val_samples   = save['my_labels']
        X_test_samples  = save['my_data']
        y_test_samples  = save['my_labels']
        del save  
        print 'Test data shape:     ', X_test_samples.shape
        print 'Test label shape:    ', y_test_samples.shape
        print 'Data successfully loaded !!'

if idisplay:
    def display_samples(num_samples=1):
        for i in range(num_samples):        
            # train samples
            idx = random.choice(range(X_train_samples.shape[0]))
            print 'Display sample train image:', idx
            plt.imshow(X_train_samples[idx].reshape(img_height,img_width), interpolation='nearest')
            plt.show()

            # test samples
            idx = random.choice(range(X_test_samples.shape[0]))
            print 'Display sample test image:', idx
            plt.imshow(X_test_samples[idx].reshape(img_height,img_width), interpolation='nearest')
            plt.show()

    display_samples()

if mnist_en:

    # params
    in_chan    = 1         # grey scale
    batch_size = 16        # batch size

    # conv1
    c1_patch_h = 3         # patch size 3x15
    c1_patch_w = 15        # patch size 3x15
    c1_depth   = 16        # 16 features (out channels)
    c1_padding = 'VALID' # padding valid
    c1_stride  = [1,1,1,1] # stride 1x1

    # maxpool1
    p1_padding = 'VALID'   # padding valid
    p1_patch   = [1,2,2,1] # patch size 2x2
    p1_stride  = [1,2,2,1] # stride 2x2
    
    # conv2
    c2_patch_h = 4         # patch size 4x20
    c2_patch_w = 20        # patch size 4x20
    c2_depth   = 32        # 32 features (out channels)
    c2_padding = 'VALID'   # padding valid
    c2_stride  = [1,1,1,1] # stride 1x1

    # maxpool2
    p2_padding = 'VALID'   # padding valid
    p2_patch   = [1,2,2,1] # patch size 2x2
    p2_stride  = [1,2,2,1] # stride 2x2

    # conv3
    c3_patch_h = 5         # patch size 5x22
    c3_patch_w = 22        # patch size 5x22
    c3_depth   = 96        # 96 features (out channels)
    c3_padding = 'VALID'   # padding valid
    c3_stride  = [1,1,1,1] # stride 1x1

    # fc
    keep_prob  = 0.9       # dropout rate
    fc_nodes   = 64        # hidden layer

    # output
    out_digits = 6         # up to 5 digits [1-5]
    out_labels = 11        # detect 0-9 & none

elif svhn_en or mydata_en:
    
    # params
    in_chan    = 1         # grey scale
    batch_size = 16        # batch size

    # conv1
    c1_patch_h = 5         # patch size 5x5
    c1_patch_w = 5         # patch size 5x5
    c1_depth   = 16        # 16 features (out channels)
    c1_padding = 'VALID'   # padding valid
    c1_stride  = [1,1,1,1] # stride 1x1

    # maxpool1
    p1_padding = 'VALID'   # padding valid
    p1_patch   = [1,2,2,1] # patch size 2x2
    p1_stride  = [1,2,2,1] # stride 2x2
    
    # conv2
    c2_patch_h = 5         # patch size 5x5
    c2_patch_w = 5         # patch size 5x5
    c2_depth   = 32        # 32 features (out channels)
    c2_padding = 'VALID'   # padding valid
    c2_stride  = [1,1,1,1] # stride 1x1

    # maxpool2
    p2_padding = 'VALID'   # padding valid
    p2_patch   = [1,2,2,1] # patch size 2x2
    p2_stride  = [1,2,2,1] # stride 2x2

    # conv3
    c3_patch_h = 5         # patch size 5x5
    c3_patch_w = 5         # patch size 5x5
    c3_depth   = 96        # 96 features (out channels)
    c3_padding = 'VALID'   # padding valid
    c3_stride  = [1,1,1,1] # stride 1x1

    # fc
    keep_prob  = 0.8       # dropout rate
    fc_nodes   = 64        # hidden layer

    # output
    out_digits = 6         # up to 5 digits [1-5]
    out_labels = 11        # detect 0-9 & none


graph = tf.Graph()
with graph.as_default():    

    # in, out place holders

    X_val  = tf.constant(X_val_samples)
    X_test = tf.constant(X_test_samples)

    Y = tf.placeholder(tf.int32, shape=(batch_size, out_digits))
    X = tf.placeholder(tf.float32, shape=(batch_size, img_height, img_width, in_chan))

    # weights & biases

    def init_bias(name, shape):
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(shape=shape, name=name, initializer=initializer)

    def init_weight(name, shape):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        return tf.get_variable(shape=shape, name=name, initializer=initializer)

    # pool_out = [(width or height) - patch]/(stride) + 1 
    # conv_out = [(width or height) - patch + 2 * pad]/(stride) + 1 
    def calc_out(w, h, p_w, p_h, stride, padding, type='conv'):
        pad = 1 if padding == 'SAME' else 0
        if type == 'pool':
            w_out = (w - p_w)/(stride) + 1
            h_out = (h - p_h)/(stride) + 1
        else:
            w_out = (w - p_w + 2 * pad)/(stride) + 1
            h_out = (h - p_h + 2 * pad)/(stride) + 1

        return w_out, h_out

    def calc_fc_size(im_w, im_h):
        (c1_w, c1_h) = calc_out(im_w, im_h, c1_patch_w,  c1_patch_h,  c1_stride[1], c1_padding, type='conv')
        (p1_w, p1_h) = calc_out(c1_w, c1_h, p1_patch[1], p1_patch[1], p1_stride[1], p1_padding, type='pool')
        (c2_w, c2_h) = calc_out(p1_w, p1_h, c2_patch_w,  c2_patch_h,  c2_stride[1], c2_padding, type='conv')
        (p2_w, p2_h) = calc_out(c2_w, c2_h, p2_patch[1], p2_patch[1], p2_stride[1], p2_padding, type='pool')
        (c3_w, c3_h) = calc_out(p2_w, p2_h, c3_patch_w,  c3_patch_h,  c3_stride[1], c3_padding, type='conv')

        print('Final image size after convolutions: {} x {}'.format(c3_w, c3_h))
        return c3_w, c3_h

    fc_w, fc_h = calc_fc_size(img_width, img_height)

    b_C1 = init_bias(name='b_C1', shape=[c1_depth])
    b_C2 = init_bias(name='b_C2', shape=[c2_depth])
    b_C3 = init_bias(name='b_C3', shape=[c3_depth])
    b_FC = init_bias(name='b_FC', shape=[fc_nodes])
        
    W_C1 = init_weight(name='W_C1', shape=[c1_patch_h, c1_patch_w, in_chan,  c1_depth])
    W_C2 = init_weight(name='W_C2', shape=[c2_patch_h, c2_patch_w, c1_depth, c2_depth])
    W_C3 = init_weight(name='W_C3', shape=[c3_patch_h, c3_patch_w, c2_depth, c3_depth])
    W_FC = init_weight(name='W_FC', shape=[fc_w * fc_h * c3_depth, fc_nodes])
        
    b_Y1 = init_bias(name='b_Y1', shape=[out_labels])
    b_Y2 = init_bias(name='b_Y2', shape=[out_labels])
    b_Y3 = init_bias(name='b_Y3', shape=[out_labels])
    b_Y4 = init_bias(name='b_Y4', shape=[out_labels])
    b_Y5 = init_bias(name='b_Y5', shape=[out_labels])
        
    W_Y1 = init_weight(name='W_Y1', shape=[fc_nodes, out_labels])
    W_Y2 = init_weight(name='W_Y2', shape=[fc_nodes, out_labels])
    W_Y3 = init_weight(name='W_Y3', shape=[fc_nodes, out_labels])
    W_Y4 = init_weight(name='W_Y4', shape=[fc_nodes, out_labels])
    W_Y5 = init_weight(name='W_Y5', shape=[fc_nodes, out_labels])
        
    # CNN Model
    def model(data, keep_prob):
        with tf.name_scope('layer_1'):
            c1_out = tf.nn.conv2d(data, W_C1, c1_stride, padding=c1_padding)
            r1_out = tf.nn.relu(c1_out + b_C1)
            p1_out = tf.nn.max_pool(r1_out, p1_patch, p1_stride, padding=p1_padding)
        
        with tf.name_scope('layer_2'):
            c2_out = tf.nn.conv2d(p1_out, W_C2, c2_stride, padding=c2_padding)
            r2_out = tf.nn.relu(c2_out + b_C2)
            p2_out = tf.nn.max_pool(r2_out, p2_patch, p2_stride, padding=p2_padding)
        
        with tf.name_scope('layer_3'):
            c3_out = tf.nn.conv2d(p2_out, W_C3, c3_stride, padding=c3_padding)
            r3_out = tf.nn.relu(c3_out + b_C3)
            d1_out = tf.nn.dropout(r3_out, keep_prob)
        
        with tf.name_scope('fc_layer'):
            shape   = d1_out.get_shape().as_list()
            reshape = tf.reshape(d1_out, [shape[0], shape[1] * shape[2] * shape[3]])
            fc_out  = tf.nn.relu(tf.matmul(reshape, W_FC) + b_FC)
        
        with tf.name_scope('fully_connected'):                
            y1 = tf.matmul(fc_out, W_Y1) + b_Y1
            y2 = tf.matmul(fc_out, W_Y2) + b_Y2
            y3 = tf.matmul(fc_out, W_Y3) + b_Y3
            y4 = tf.matmul(fc_out, W_Y4) + b_Y4
            if predict_bbox:
                y5 = 0
            else:
                y5 = tf.matmul(fc_out, W_Y5) + b_Y5

        return [y1, y2, y3, y4, y5]

    # Loss function: cross_entropy 
    [y1, y2, y3, y4, y5] = model(X, keep_prob)

    with tf.name_scope("cross_entropy"):        
        if prdict_bbox:
            # regression loss
            cross_entropy =  \
                             tf.reduce_mean(tf.square(y1 - Y[:,1])) +\
                             tf.reduce_mean(tf.square(y2 - Y[:,2])) +\
                             tf.reduce_mean(tf.square(y3 - Y[:,3])) +\
                             tf.reduce_mean(tf.square(y4 - Y[:,4]))
        else:
            cross_entropy = \
                            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y1, Y[:, 1])) + \
                            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y2, Y[:, 2])) + \
                            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y3, Y[:, 3])) + \
                            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y4, Y[:, 4])) + \
                            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y5, Y[:, 5]))
        tf.summary.scalar("cross_entropy", cross_entropy)

    # Optimizer
    alpha = 0.05; learn_step = tf.Variable(0)
    learn = tf.train.exponential_decay(alpha, learn_step, 10000, 0.96)
    optimizer = tf.train.AdagradOptimizer(learn).minimize(cross_entropy, global_step=learn_step)

    def output_combine(data):
        if predict_bbox:
            y = tf.pack([model(data, 1.0)[0],
                         model(data, 1.0)[1],
                         model(data, 1.0)[2],
                         model(data, 1.0)[3],
                         model(data, 1.0)[4]])
        else:
            y = tf.pack([
                tf.nn.softmax(model(data, 1.0)[0]),
                tf.nn.softmax(model(data, 1.0)[1]),
                tf.nn.softmax(model(data, 1.0)[2]),
                tf.nn.softmax(model(data, 1.0)[3]),
                tf.nn.softmax(model(data, 1.0)[4])])
        return y

    y_pred      = output_combine(X)
    y_val_pred  = output_combine(X_val)
    y_test_pred = output_combine(X_test)
        
    # Save
    saver = tf.train.Saver()

    # weight histogram
    tf.summary.histogram("W_C1", W_C1)
    tf.summary.histogram("W_C2", W_C2)
    tf.summary.histogram("W_C3", W_C3)
    tf.summary.histogram("W_FC", W_FC)
    tf.summary.histogram("W_Y1", W_Y1)
    tf.summary.histogram("W_Y2", W_Y2)
    tf.summary.histogram("W_Y3", W_Y3)
    tf.summary.histogram("W_Y4", W_Y4)
    tf.summary.histogram("W_Y5", W_Y5)

    tf.summary.histogram("b_C1", b_C1)
    tf.summary.histogram("b_C2", b_C2)
    tf.summary.histogram("b_C3", b_C3)
    tf.summary.histogram("b_FC", b_FC)
    tf.summary.histogram("b_Y1", b_Y1)
    tf.summary.histogram("b_Y2", b_Y2)
    tf.summary.histogram("b_Y3", b_Y3)
    tf.summary.histogram("b_Y4", b_Y4)
    tf.summary.histogram("b_Y5", b_Y5)

    print('Graph done!')

#%%
        
def accuracy(predictions, labels, debug=0):
    if debug:
        for i in range(labels.shape[0]):
            print 'Test ',i+1,':', np.argmax(predictions, 2).T[i], labels[i]

    return (100.0 * np.sum(np.argmax(predictions, 2).T == labels)
             / predictions.shape[1] / predictions.shape[0])

def get_offset(step, batch_size, data):
    offset = (step * batch_size) % (data.shape[0] - batch_size)
    return offset

def model_loop(X_samples, y_samples, num_steps=1, debug=0):
    for step in range(num_steps):
        offset  = get_offset(step, batch_size, y_samples)
        batch_X = X_samples[offset:(offset + batch_size), :, :, :]
        batch_Y = y_samples[offset:(offset + batch_size), :]        
        feed_dict = {X: batch_X, Y: batch_Y}

        _, loss, pred, summary = sess.run([optimizer, cross_entropy, y_pred, merged], feed_dict=feed_dict)

        writer.add_summary(summary)

        if (step % 250 == 0):
            print (('step {}: loss -> {} accuracy -> {}%').format(step, round(loss,2), accuracy(pred, batch_Y[:,1:6], debug=debug)))
            print (('Validation accuracy: {}%'.format(round(accuracy(y_val_pred.eval(), y_val_samples[:,1:6], debug=debug), 2))))

with tf.Session(graph=graph) as sess:
    writer = tf.summary.FileWriter("log", sess.graph)
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    
    if restore_session:
        print('Restoring session: ', session_name)
        saver.restore(sess, session_name)
    else:
        # train loops
        print ('Start training: batch_size {} num_steps {}').format(batch_size, num_steps)
        model_loop(X_train_samples, y_train_samples, num_steps, debug=0)

    # test accuracy
    print ('Start Testing: num_tests {}').format(num_tests)
    print (('Test accuracy: {}%'.format(accuracy(y_test_pred.eval(), y_test_samples[:,1:6], debug=debug))))

    # save session
    save_path = saver.save(sess, "session/digit_recognizer.ckpt")
    print('Model saved to file: {}'.format(save_path))

print('Tensorboard: tensorboard --logdir=log')
