########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names
import os
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import csv
import re
import nltk
import json
import urllib2
import os
from pathlib import Path
import PIL
from PIL import Image
import tensorflow as tf
import numpy as np

warnings.filterwarnings("ignore")




class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))




f_obj = open("Jeans.csv")
reader = csv.DictReader(f_obj, delimiter=',')

productId = []
title = []
description = []
imageUrlStr2 = []
mrp = []
sellingPrice = []
specialPrice = []
FlipkartLink = []
productBrand = []
inStock = []
codAvailable = []
discount = []
shippingCharges = []
size = []
color = []
keySpecsStr = []
detailedSpecsStr = []
specificationList = []
sellerName = []
sellerAverageRating = []
sellerNoOfRatings = []
sellerNoOfReviews = []

for line in reader:
    productId.append(line["productId"])
    title.append(line["title"].lower())
    description.append(line["description"].lower())
    imageUrlStr2.append(line["imageUrlStr2"].lower())
    mrp.append(line["mrp"].lower())
    sellingPrice.append(line["sellingPrice"].lower())
    specialPrice.append(line["specialPrice"].lower())
    FlipkartLink.append(line["FlipkartLink"].lower())
    productBrand.append(line["productBrand"].lower())
    inStock.append(line["inStock"].lower())
    codAvailable.append(line["codAvailable"].lower())
    discount.append(line["discount"].lower())
    shippingCharges.append(line["shippingCharges"].lower())
    size.append(line["size"].lower())
    color.append(line["color"].lower())
    keySpecsStr.append(line["keySpecsStr"].lower())
    detailedSpecsStr.append(line["detailedSpecsStr"].lower())
    specificationList.append(line["specificationList"].lower())
    sellerName.append(line["sellerName"].lower())    
    sellerAverageRating.append(line["sellerAverageRating"].lower())
    sellerNoOfRatings.append(line["sellerNoOfRatings"].lower())
    sellerNoOfReviews.append(line["sellerNoOfReviews"].lower())

#print sellerName[:5]

no_of_rows = len(productId)
print "no_of_rows : ", no_of_rows


SpecsCombined = []
for i in range(no_of_rows):
    SpecsCombined.append(re.sub(r'Fit:',r'',re.sub(r'Fabric:',r'',keySpecsStr[i])) + " " +
                         re.sub(r'Fit:',r'',re.sub(r'Fabric:',r'',detailedSpecsStr[i])) + " " + 
                         re.sub(r'Fit:',r'',re.sub(r'Fabric:',r'',specificationList[i])))

for i in range(no_of_rows):
    description[i] = re.sub(r'[^\w\s]', ' ',description[i]).lower()

description_words = [word for line in description for word in line.strip().split()]


stop_words_ = [
'a','above','again','against','am','an','are','and','for','in','if','or','as','is'
'arent','because','been','being','below',
'between','both','but','by','during','each',
'few','further','had','had','has','hasnt','have','havent',
'having','he','hed','hell','hes','her','here','heres',
'hers','herself','him','himself','his','i','ill','im',
'ive','into','isnt','it','its','its','itself','me',
'more','my','myself','no','nor','of','off',
'on','once','only','other','our','ours','ourselves','out',
'over','own','same','shant','she','shed','she','she',
'so','some','such','than','that','thats','the','their',
'theirs','them','themselves','then','there','theres','these','they',
'theyd','theyll','theyre','theyve','this','those','through','to',
'too','under','up','very','was','wasnt','we',
'wed','well','were','weve','were','werent','what','whats',
'where','wheres','which','who','whos','whom','why','whys',
'youd','youll','youre','youve','your','yours','yourself','yourselves','with','is','you']

description_words_except_stop_words = []
for item in description_words:
  if item not in stop_words_ and len(item) > 1:
    description_words_except_stop_words.append(item)


print "len(description_words_except_stop_words) : ",len(description_words_except_stop_words)
description_words_except_stop_words = nltk.FreqDist(description_words_except_stop_words)
#print "\nMost common 30 words are : ", description_words_except_stop_words.most_common(30)

word_features = [tup[0] for tup in description_words_except_stop_words.most_common(500)]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

    
#print((find_features(description[0].split())))
#outfile = open('data.json', 'w')

#json.dump(find_features(description[0].split()), outfile,indent=4, sort_keys=True)

out_ = {}

'''
#CODE THAT DOWNLOADES IMAGES TO ../IMAGES
for i in range(no_of_rows):
    url = imageUrlStr2[i]
    file_name = url.split('/')[-1]
    u = urllib2.urlopen(url)
    f = open(os.path.join('../images', str(i)+'.jpg'), 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print i,") Downloading: %s Bytes: %s" % (file_name, file_size)
    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break
        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,

    f.close()
'''
if __name__ == '__main__':
    '''
    #CODE USED TO EXTRACT FEATURE VECTORS OF IMAGES AND SAVE THEM, THUS IT HAS BEEN COMMENTED NOW
    #IT USES VGG-16 PRE TRAINED MODEL TO EXTRACT FEATURES

    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    images = []
    data_dir = '../images'
    count = 0
    
    # DUE TO RAM CONSTRAINTS ALL IMAGE DATA COULD NOT BE CONVERTED TO FEATURES ALL AT ONCE, THEREFORE
    # THIS PART IS RUN MULTIPLE TIMES, CONVERTING 30 IMAGES AT ONCE AND SAVING THEM,

    for i in range(210,240):
        if os.path.exists(data_dir + '/' + str(i) + '.jpg')  == False:
            continue
        count += 1
        img = imread(data_dir + '/' + str(i) + '.jpg')
        img = imresize(img,(224, 224))
        images.append(img)

    print len(images)

    features = sess.run(vgg.fc2, feed_dict={vgg.imgs: images})
    
    print "features.size : ",features[:,0].size," ",features[0,:].size

    np.savez('features210toend.npz',a=features)
    print " Feature extraction done"
    '''    

    data = np.load('features0to29.npz')
    features = data['a']
    print "0 to 29 loaded"
    data = np.load('features30to59.npz')
    features = np.concatenate((features,data['a']))
    print "30 to 59 loaded"
    data = np.load('features60to89.npz')
    features = np.concatenate((features,data['a']))
    print "60 to 89 loaded"
    data = np.load('features90to119.npz')
    features = np.concatenate((features,data['a']))
    print "90 to 119 loaded"
    data = np.load('features120to149.npz')
    features = np.concatenate((features,data['a']))
    print "120 to 149 loaded"
    data = np.load('features150to179.npz')
    features = np.concatenate((features,data['a']))
    print "150 to 179 loaded"
    data = np.load('features180to209.npz')
    features = np.concatenate((features,data['a']))
    print "180 to 209 loaded"
    data = np.load('features210toend.npz')
    features = np.concatenate((features,data['a']))
    print "210 to end loaded"

    features = np.asarray(features)
    print features[:,0].size," ",features[0,:].size

    # COMPUTING COSINE SIMILARITY OF VECTORES GENERATED FROM IMAGE BY VGG-16
    sim = np.zeros((features[:,0].size,features[:,0].size))
    for i in range(features[:,0].size):
        for j in range(i+1,features[:,0].size):
            sim[i,j] = cosine_similarity(features[i], features[j])

    np.savez('similarity.npz',a=sim)        
    
    data = np.load('similarity.npz')
    sim = data['a']

    out_ = {}

    count_same = 0
    no_of_rows = features[:,0].size

    for i in range(no_of_rows):
        for j in range(i+1,no_of_rows):
            if set(title[i].split(' ')) != set(title[j].split(' ')):
                continue
            if set(color[i].split(' ')) != set(color[j].split(' ')):
                continue
            if set(mrp[i].split(' ')) != set(mrp[j].split(' ')):
                continue
            if set(productBrand[i].split(' ')) != set(productBrand[j].split(' ')):
                continue
            if set(SpecsCombined[i].split(' ')) != set(SpecsCombined[j].split(' ')):
                continue
            if set(sellerName[i].split(' ')) != set(sellerName[j].split(' ')):
                continue
            # CONDITIONS FOR DUPLICATE ARE, SAME TITLE,COLOR,MRP,PRODUCT BRAND, COMBINED SPECS, SELLER NAME
            # AND MOST IMPORTANTLY COSINE SIMILARITY GREATER THAN 0.9    
            if sim[i,j] > 0.90:
                print "i,j = ",i," ",j," and sim = ",sim[i,j]
                count_same += 1
                if productId[i] not in out_:
                    out_[productId[i]] = []
                    out_[productId[i]].append((productId[j],sim[i,j]))    
                else:
                    out_[productId[i]].append((productId[j],sim[i,j]))    

    outfile = open('similarity.json', 'w')

    json.dump(out_, outfile,indent=4, sort_keys=True)
    outfile.close()

    print "count_same : ",count_same