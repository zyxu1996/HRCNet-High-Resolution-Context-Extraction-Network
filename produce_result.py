import numpy as np
import os
import scipy.io as io
import matplotlib.pyplot as plt
import skimage.io as skimage_io
import scipy.misc


base_dataset_dir_voc = './dataset/'


train_images_folder_name_voc = 'images/train/'
train_annotations_folder_name_voc = 'images/label/'
train_images_dir_voc = os.path.join(base_dataset_dir_voc,train_images_folder_name_voc)
train_annotations_dir_voc = os.path.join(base_dataset_dir_voc, train_annotations_folder_name_voc)

val_images_folder_name_voc = 'images/test/'
val_annotations_folder_name_voc = 'labels/test/'

val_images_dir_voc = os.path.join(base_dataset_dir_voc, val_images_folder_name_voc)
val_annotations_dir_voc = os.path.join(base_dataset_dir_voc, val_annotations_folder_name_voc)

#读取文件列表
train_images_filename_list= os.listdir(train_images_dir_voc)
val_images_filename_list = os.listdir(val_images_dir_voc)

train_list = ['top_potsdam_2_10', 'top_potsdam_2_11', 'top_potsdam_2_12', 'top_potsdam_3_10',
              'top_potsdam_3_11', 'top_potsdam_3_12', 'top_potsdam_4_10', 'top_potsdam_4_11',
              'top_potsdam_4_12', 'top_potsdam_5_10', 'top_potsdam_5_11', 'top_potsdam_5_12',
              'top_potsdam_6_10', 'top_potsdam_6_11', 'top_potsdam_6_12', 'top_potsdam_6_7',
              'top_potsdam_6_8', 'top_potsdam_6_9', 'top_potsdam_7_10', 'top_potsdam_7_11',
              'top_potsdam_7_12', 'top_potsdam_7_7', 'top_potsdam_7_8', 'top_potsdam_7_9', ]

test_list = ['top_potsdam_2_13', 'top_potsdam_2_14', 'top_potsdam_3_13', 'top_potsdam_3_14',
             'top_potsdam_4_13', 'top_potsdam_4_14', 'top_potsdam_4_15', 'top_potsdam_5_13',
             'top_potsdam_5_14', 'top_potsdam_5_15', 'top_potsdam_6_13', 'top_potsdam_6_14',
             'top_potsdam_6_15', 'top_potsdam_7_13']

def label_to_RGB(image):
    RGB = np.zeros(shape=[image.shape[0], image.shape[1], 3], dtype=np.uint8)
    index = image == 0
    RGB[index] = np.array([255, 255, 255])
    index = image == 1
    RGB[index] = np.array([0, 0, 255])
    index = image == 2
    RGB[index] = np.array([0, 255, 255])
    index = image == 3
    RGB[index] = np.array([0, 255, 0])
    index = image == 4
    RGB[index] = np.array([255, 255, 0])
    index = image == 5
    RGB[index] = np.array([255, 0, 0])
    return RGB


# for image_name in test_list:
# # for image_name in train_list: #test train_dataset
#     w_patch = []
#     h_patch = []
#     for i in range(11):
#         for j in range(11):
#             # name = image_name + '_' + str(i*12+j)
#             # patch = scipy.misc.imread('../dataset/dataset/5_Labels_all_1D_train/' + name + '.png')  #test train_dataset
#
#             # name = image_name + '_' + str(i * 11 + j)
#             # name = image_name + '_' + str(i * 19 + j)   #test_test384_72_dataset
#             # patch = scipy.misc.imread('../dataset/dataset/5_Labels_all_1D_train_384_72/' + name + '.png')  #test test_dataset
#
#             name = image_name + '_' + str(i * 11 + j) + '.mat'
#             location = val_images_filename_list.index(name)
#             patch = io.loadmat('network_output/' + str(location+1) + '.mat')['network_output']
#             if j == 0:
#                 w_patch = patch
#             else:
#                 w_patch = np.concatenate((w_patch, patch), axis=1)
#                 # w_patch = np.concatenate((w_patch, patch[:, 64:512]), axis=1)  #test train_dataset
#                 # w_patch = np.concatenate((w_patch, patch[:, 72:384]), axis=1)  #test test384_72_dataset
#         if i == 0:
#             h_patch = w_patch
#         else:
#             h_patch = np.concatenate((h_patch, w_patch), axis=0)
#             # h_patch = np.concatenate((h_patch, w_patch[64:512, :]), axis=0)  #test train_dataset
#             # h_patch = np.concatenate((h_patch, w_patch[72:384, :]), axis=0)  #test test384_72_dataset
#     image = h_patch
#
# "test nostride 384_72"
# for image_name in test_list:
#     # for image_name in train_list: #test train_dataset
#     w_patch = []
#     h_patch = []
#     for i in range(16):
#         for j in range(16):
#
#             name = image_name + '_' + str(i * 16 + j) + '.mat'
#             location = val_images_filename_list.index(name)
#             patch = io.loadmat('network_output/' + str(location + 1) + '.mat')['network_output']
#             if j == 0:
#                 w_patch = patch
#             elif j < 15:
#                 w_patch = np.concatenate((w_patch, patch), axis=1)
#             elif j == 15:
#                 w_patch = np.concatenate((w_patch, patch[:, 144:384]), axis=1)
#         if i == 0:
#             h_patch = w_patch
#         elif i < 15:
#             h_patch = np.concatenate((h_patch, w_patch), axis=0)
#         elif i == 15:
#             h_patch = np.concatenate((h_patch, w_patch[144:384, :]), axis=0)
#     image = h_patch
#     "test nostride 384_72"

"test 384_192"
for image_name in test_list:
    # for image_name in train_list: #test train_dataset
    w_patch = []
    h_patch = []
    for i in range(31):
        for j in range(31):

            name = image_name + '_' + str(i * 31 + j) + '.mat'
            location = val_images_filename_list.index(name)
            patch = io.loadmat('network_output/' + str(location + 1) + '.mat')['network_output']
            if j == 0:
                w_patch = patch[:, 0:288]
            elif j < 30:
                w_patch = np.concatenate((w_patch, patch[:, 96:288]), axis=1)
            elif j == 30:
                w_patch = np.concatenate((w_patch, patch[:, 240:384]), axis=1)
        if i == 0:
            h_patch = w_patch[0:288, :]
        elif i < 30:
            h_patch = np.concatenate((h_patch, w_patch[96:288, :]), axis=0)
        elif i == 30:
            h_patch = np.concatenate((h_patch, w_patch[240:384, :]), axis=0)
    image = h_patch
    "test 384_192"

    scipy.misc.toimage(image, cmin=0, cmax=255).save('network_result_1D/' + image_name + '.png')
    plt.imshow(image)
    plt.show()
    image_RGB = label_to_RGB(image)
    skimage_io.imsave('network_result/' + image_name + '.tif', image_RGB)
    plt.imshow(image_RGB)
    plt.show()



