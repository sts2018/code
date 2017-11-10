#
# https://github.com/kratzert/finetune_alexnet_with_tensorflow/blob/master/validate_alexnet_on_imagenet.ipynb
#
#some basic imports and setups
import getopt
import sys
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import manipulate as ma

tf.reset_default_graph()

# mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
sigma = 0
use_exist = 0

test_dir = 'test'
test_file = ''

current_dir = os.getcwd()
checkpoint_path = os.path.join(current_dir, 'ckp_star')
checkpoint_name = os.path.join(checkpoint_path, 'model_epoch10.ckpt')

do_bn = False

try:
    opts, args = getopt.getopt(sys.argv[1:],"bset:f:c:")
except getopt.GetoptError:
   print '-t'
   sys.exit(2)

for opt, arg in opts:
    if opt in '-b':
        do_bn = True
    elif opt in '-t':
        test_dir = arg
    elif opt in '-f':
        test_file = arg
    elif opt in '-s':
        sigma = 1
    elif opt in '-e':
        use_exist = 1
    elif opt in '-c':
        checkpoint_name = arg
    else:
        print '-t'
        sys.exit(2)

if test_file == '':
    image_dir = os.path.join(current_dir, test_dir)
    # convert gif to jpg
    orig_img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.gif')]
else:
    orig_img_files = []
    f = open(test_file, 'r')
    lines = f.readlines()
    for line in lines:
        items = line.split(' ')
        if (items[0][-8:] != "q100.jpg"): # hack - keep the images ending with q100.jpg only
            continue
        orig_img_files.append(items[0])
      # append(int(items[1]))

img_files = []
for img in orig_img_files:
    outfile = os.path.splitext(img)[0]+"_q100.jpg"
    if (use_exist == 0):
        Image.open(img).convert('RGB').save(outfile, subsampling=0, quality=100)
        ma.sigmaimage2(outfile, sigma)
    img_files.append(outfile)

# load all images
# imgs = []
# for f in img_files:
#    imgs.append(cv2.imread(f))

from alexnet import AlexNet
from astro_classes import class_names

#placeholder for input and dropout rate
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

#create model with default config ( == no skip_layer and 1000 units in the last layer)
#model = AlexNet(x, keep_prob, 1000, [])
#why?
num_classes = len(class_names)
model = AlexNet(x, keep_prob, num_classes, [], do_bn, False)

#define activation of last layer as score
score = model.fc8

#create op to calculate softmax 
softmax = tf.nn.softmax(score)

#retrieve checkpoint
saver = tf.train.Saver()

stats = {}
for class_name in class_names:
    stats[class_name] = 0

with tf.Session() as sess:
    
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    # Load the pretrained weights into the model
    # model.load_initial_weights(sess)

    # Load checkpoint
    saver.restore(sess, checkpoint_name)
    
    # Create figure handle
    # fig2 = plt.figure(figsize=(15,6))
    
    # Loop over all images
    for i, img_file in enumerate(img_files):
        
        image = cv2.imread(img_file)

        # Convert image to float32 and resize to (227x227)
        img = cv2.resize(image.astype(np.float32), (227,227))
        
        # Subtract the ImageNet mean
        img -= imagenet_mean
        
        # Reshape as needed to feed into model
        img = img.reshape((1,227,227,3))
        
        # Run the session and calculate the class probability
        probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})
        
        # Get the class name of the class with the highest probability
        class_name = class_names[np.argmax(probs)]
        print img_files[i], "Class: ", class_name, ", probability: ", probs[0,np.argmax(probs)]
        stats[class_name] = stats[class_name] + 1

        # Plot image with class name and prob in the title
        # fig2.add_subplot(1,3,i+1)
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.title("Class: " + class_name + ", probability: %.4f" %probs[0,np.argmax(probs)])
        # plt.axis('off')

    print stats
