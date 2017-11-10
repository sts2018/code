import getopt
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import PIL
from PIL import Image
import manipulate as ma

def augment_image_files(img_files, augment, sigma):

    outfiles = []

    # Not augment the image files
    if (augment == 0):
        for img in img_files:
            outfile = os.path.splitext(img)[0]+"_q100.jpg"
            Image.open(img).convert('RGB').save(outfile, subsampling=0, quality=100)
            ma.sigmaimage2(outfile, sigma)
            outfiles.append(outfile)
        return outfiles

    # Augment the image files
    for img in img_files:
        a_list = []
        # outfile = os.path.splitext(img)[0]+"_q60.jpg"
        # Image.open(img).convert('RGB').save(outfile, subsampling=0, quality=60)
        # ma.sigmaimage2(outfile, sigma)
        # a_list.append(outfile)
        outfile = os.path.splitext(img)[0]+"_q100.jpg"
        Image.open(img).convert('RGB').save(outfile, subsampling=0, quality=100)
        ma.sigmaimage2(outfile, sigma)
        a_list.append(outfile)
        outfile = os.path.splitext(img)[0]+"_fliplr.jpg"
        Image.open(img).transpose(PIL.Image.FLIP_LEFT_RIGHT).convert('RGB').save(outfile, subsampling=0, quality=100)
        ma.sigmaimage2(outfile, sigma)
        a_list.append(outfile)
        # outfile = os.path.splitext(img)[0]+"_fliptb.jpg"
        # Image.open(img).transpose(PIL.Image.FLIP_TOP_BOTTOM).convert('RGB').save(outfile, subsampling=0, quality=100)
        # ma.sigmaimage2(outfile, sigma)
        # a_list.append(outfile)

        # Rotate images - these are already sigma processed
        for img in a_list:
            f = cv2.imread(img)
            image_height, image_width = f.shape[0:2]

            for i in np.arange(15, 360, 15):
                f_orig = np.copy(f)
                f_rotated = ma.rotate_image(f, i)
                f_rotated_cropped = ma.crop_around_center(
                    f_rotated,
                    *ma.largest_rotated_rect(
                        image_width,
                        image_height,
                        math.radians(i)
                        )
                    )
                outfile = os.path.splitext(img)[0]+'_r'+str(i)+".jpg"
                cv2.imwrite(outfile, f_rotated_cropped)
                outfiles.append(outfile)

        outfiles.extend(a_list)
    return outfiles

from random import seed,shuffle
from math import floor

def split_list(list, split_perc):
    seed(0)
    shuffle(list)
    split_perc = split_perc
    split_index = int(floor(len(list) * split_perc))
    list_1 = list[:split_index]
    list_2 = list[split_index:]
    return (list_1, list_2)


train_filename = 'train.txt'
val_filename = 'val.txt'

def update_train_file(file, img_list, cat):
    target = open(file, 'a')
    for img in img_list:
        target.write(img+' '+str(cat))
        target.write("\n")


def update_class_names_file(names):
    target = open('astro_classes.py', 'w')
    target.write("class_names = ")
    target.write("[")
    for name in names:
        target.write("'"+name+"',")
    target.write("]")


def process_image_files(image_dir, cat, augment, split_perc, sigma):

    img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
    for img in img_files:
        os.remove(img)
    img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.gif')]
    print 'image count:', len(img_files)
    (train_list, val_list) = split_list(img_files, split_perc)

    list = augment_image_files(train_list, augment, sigma)
    update_train_file(train_filename, list, cat)

    list = augment_image_files(val_list, augment, sigma)
    update_train_file(val_filename, list, cat)


#
# Main
#
try:
    opts, args = getopt.getopt(sys.argv[1:],"t:asp:")
except getopt.GetoptError:
   print '-f'
   sys.exit(2)

target = "fri,frii"
augment = 0
sigma = 0
cat = ''
split_perc = 0.7

for opt, arg in opts:
    if opt in '-t':
        target = arg
    elif opt in '-a':
        augment = 1
    elif opt in '-s':
        sigma = 1
    elif opt in '-p':
        split_perc = float(arg)
    else:
        print '-t:ap:'
        sys.exit(2)

if os.path.exists(train_filename):
    os.remove(train_filename)
if os.path.exists(val_filename):
    os.remove(val_filename)

current_dir = os.getcwd()
class_names = target.split(',')
update_class_names_file(class_names)

print '---------------------------------------'
print 'Config:'
print '  Target....: ', target
print '  Augment..." ', augment
print '  Sigma.....: ', sigma
print '  Split.....: ', split_perc
print '---------------------------------------'

count = 0
for cname in class_names:
    image_dir = os.path.join(current_dir, cname)
    print image_dir, count
    process_image_files(image_dir, count, augment, split_perc, sigma)
    count = count + 1

print 'training: ', sum(1 for line in open(train_filename))
print 'val:', sum(1 for line in open(val_filename))























    
