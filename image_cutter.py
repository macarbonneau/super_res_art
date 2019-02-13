import os
import dataset
import random
import numpy as np


in_path = "/home/randysavage/projects/examples/super_resolution/dataset/Laurent"
image_path = os.path.join(in_path, 'images')

original_path = os.path.join(image_path, 'original')

train_path = os.path.join(image_path, 'train')
train_list = os.path.join(in_path, 'iids_train.txt')
test_path = os.path.join(image_path, 'test')
test_list = os.path.join(in_path, 'iids_test.txt')

sub_image_size = 256
nb_sub_image_per_image_train = 500
nb_sub_image_per_image_test = 500

# with open(train_list, 'w+') as image_file:
#     print('file created')

count = 0
for image_file in os.listdir(original_path):

    if dataset.is_image_file(image_file):
        im = dataset.load_img(os.path.join(original_path, image_file))

        for i in range(nb_sub_image_per_image_train):

            while True:
                start_x = random.randint(0, im.size[0] - sub_image_size)
                start_y = random.randint(0, im.size[1] - sub_image_size)
                bbox = (start_x, start_y, start_x + sub_image_size, start_y + sub_image_size)

                sub_im = im.crop(bbox)
                a = np.array(sub_im)
                s = np.sum(a[:])
                if s > sub_image_size**2*5:
                    file_name = str(count) + '_' + str(s) + '.png'
                    sub_im.save(os.path.join(train_path, file_name))
                    with open(train_list, 'a+') as f:
                        f.write(str(file_name) + '\n')
                    count += 1
                    break

        for i in range(nb_sub_image_per_image_test):

            while True:
                start_x = random.randint(0, im.size[0] - sub_image_size)
                start_y = random.randint(0, im.size[1] - sub_image_size)
                bbox = (start_x, start_y, start_x + sub_image_size, start_y + sub_image_size)

                sub_im = im.crop(bbox)
                a = np.array(sub_im)
                s = np.sum(a[:])
                if s > sub_image_size**2*5:
                    file_name = str(count) + '_' + str(s) + '.png'
                    sub_im.save(os.path.join(test_path, file_name))
                    with open(test_list, 'a+') as f:
                        f.write(str(file_name) + '\n')
                    count += 1
                    break




