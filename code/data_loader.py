import os
import cv2
import h5py
import torch
import random
import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter

# Class to fetch and process RMFD data
class object_data():
    def __init__(self, data_dir, batch_size=50, img_resize_size=(100,100)):
        super(object_data, self).__init__()
        self.data_dir = data_dir                    # Directory of the data           
        self.batch_size = batch_size                # Batch size
        self.img_resize_size = img_resize_size      # Size of the image after resize -> (width, height)
        self.test_data_size = 500                   # Reserves this much number of samples from each class for testing
        
        self.test_X = None                          # Stores test data
        self.test_Y = None                          # Stores test labels

        self.masked_dir_path = os.path.join(self.data_dir, 'AFDB_masked_face_dataset')  # Path to the directory containing masked faces
        self.unmasked_dir_path = os.path.join(self.data_dir, 'AFDB_face_dataset')       # Path to the directory containing unmasked faces

    def prepare_data(self):                   # Preparing train and test data and returning their paths
        if not os.path.isdir(os.path.join(self.data_dir, "HDF5_version")):
            os.makedirs(os.path.join(self.data_dir, "HDF5_version"))

        pth_mask_tr = os.path.join(self.data_dir, "HDF5_version", "image_files_masked_train.h5")    # Path to HDF5 training masked images
        pth_umsk_tr = os.path.join(self.data_dir, "HDF5_version", "image_files_unmasked_train.h5")  # Path to HDF5 training unmasked images

        pth_ts = os.path.join(self.data_dir, "HDF5_version", "image_files_test.h5")     # Path to HDF5 testing masked and unmasked images

        if not (os.path.exists(pth_mask_tr) and os.path.exists(pth_umsk_tr) and os.path.exists(pth_ts)):   # Checking whether preprocessed HDF5 file is present
            # Creating HDF5 files
            mask_tr_file = h5py.File(pth_mask_tr, 'w')  # For Masked Training files
            umsk_tr_file = h5py.File(pth_umsk_tr, 'w')  # For Unmasked Training files
            ts_file = h5py.File(pth_ts, 'w')            # For Masked and Unmasked Testing files

            mask_filepaths, umsk_filepaths = [], [] # Lists storing filepaths

            # Reading the masked filepaths
            for items in os.listdir(self.masked_dir_path):
                temp1 = os.path.join(self.masked_dir_path, items)
                if os.path.isdir(temp1):
                    for img in os.listdir(temp1):
                        temp2 = os.path.join(self.masked_dir_path, items, img)
                        if os.path.isfile(temp2):
                            mask_filepaths.append(temp2)

            # Reading the unmasked filepaths
            for items in os.listdir(self.unmasked_dir_path):
                temp1 = os.path.join(self.unmasked_dir_path, items)
                if os.path.isdir(temp1):
                    for img in os.listdir(temp1):
                        temp2 = os.path.join(self.unmasked_dir_path, items, img)
                        if os.path.isfile(temp2):
                            umsk_filepaths.append(temp2)

            # Filepath size limiter (for smaller unmasked dataset)
            min_size = min(3*len(mask_filepaths), len(umsk_filepaths))
            umsk_filepaths = umsk_filepaths[:min_size]

            # Image counts
            mask_img_count, umsk_img_count = len(mask_filepaths), len(umsk_filepaths)

            # Note: Labels -> 0: Unmasked, 1: Masked

            # Creating space in HDF5 file for Masked Training Images
            x_img_mask_tr = mask_tr_file.create_dataset('mask_images', shape=(mask_img_count-self.test_data_size, self.img_resize_size[0], self.img_resize_size[1], 3), dtype=np.float32)
            y_lab_mask_tr = mask_tr_file.create_dataset('mask_labels', shape=(mask_img_count-self.test_data_size,), dtype=np.float32)

            # Creating space in HDF5 file for Unmasked Training Images
            x_img_umsk_tr = umsk_tr_file.create_dataset('umsk_images', shape=(umsk_img_count-self.test_data_size, self.img_resize_size[0], self.img_resize_size[1], 3), dtype=np.float32)
            y_lab_umsk_tr = umsk_tr_file.create_dataset('umsk_labels', shape=(umsk_img_count-self.test_data_size,), dtype=np.float32)

            # Creating space in HDF5 file for Masked and Unmasked Testing Images
            x_img_ts = ts_file.create_dataset('test_images', shape=(2*self.test_data_size, self.img_resize_size[0], self.img_resize_size[1], 3), dtype=np.float32)
            y_lab_ts = ts_file.create_dataset('test_labels', shape=(2*self.test_data_size,), dtype=np.float32)
            
            # Storing images into HDF5 file for Masked and Unmasked Testing Images
            counter = 0
            for i in range(self.test_data_size):
                x_img_ts[counter] = cv2.cvtColor(cv2.resize(cv2.imread(mask_filepaths[i]), self.img_resize_size), cv2.COLOR_BGR2RGB).astype(np.float32)
                y_lab_ts[counter] = 1.0
                counter += 1

                x_img_ts[counter] = cv2.cvtColor(cv2.resize(cv2.imread(umsk_filepaths[i]), self.img_resize_size), cv2.COLOR_BGR2RGB).astype(np.float32)
                y_lab_ts[counter] = 0.0
                counter += 1

            # Storing images into HDF5 file for Masked Training Images
            counter = 0
            for i in range(self.test_data_size, mask_img_count):
                x_img_mask_tr[counter] = cv2.cvtColor(cv2.resize(cv2.imread(mask_filepaths[i]), self.img_resize_size), cv2.COLOR_BGR2RGB).astype(np.float32)
                y_lab_mask_tr[counter] = 1.0
                counter += 1

            # Storing images into HDF5 file for Unmasked Training Images
            counter = 0
            for i in range(self.test_data_size, umsk_img_count):
                x_img_umsk_tr[counter] = cv2.cvtColor(cv2.resize(cv2.imread(umsk_filepaths[i]), self.img_resize_size), cv2.COLOR_BGR2RGB).astype(np.float32)
                y_lab_umsk_tr[counter] = 0.0
                counter += 1

            # Closing files
            mask_tr_file.close()
            umsk_tr_file.close()
            ts_file.close()
        
        return pth_mask_tr, pth_umsk_tr, pth_ts     # Returning paths

    def fetch_train_data(self):                             # Reads training data
        pth_mask_tr, pth_umsk_tr, _ = self.prepare_data()   # Preparing data if already not done

        # Reading HDF5 files
        mask_tr_file = h5py.File(pth_mask_tr, 'r')          # For Masked Training files
        umsk_tr_file = h5py.File(pth_umsk_tr, 'r')          # For Unmasked Training files

        # Extracting images and labels from all the HDF5 files
        x_img_mask_tr = mask_tr_file['mask_images']         # Masked Training
        y_lab_mask_tr = mask_tr_file['mask_labels']

        x_img_umsk_tr = umsk_tr_file['umsk_images']         # Unmasked Training
        y_lab_umsk_tr = umsk_tr_file['umsk_labels']

        return x_img_mask_tr, y_lab_mask_tr, x_img_umsk_tr, y_lab_umsk_tr
    
    def fetch_test_data(self):              # Reads test data
        _, _, pth_ts = self.prepare_data()  # Preparing data if already not done

        # Reading HDF5 files
        ts_file = h5py.File(pth_ts, 'r')    # For Masked and Unmasked Testing files

        # Extracting images and labels from all the HDF5 files
        x_img_ts = ts_file['test_images']   # Test
        y_lab_ts = ts_file['test_labels']

        return x_img_ts, y_lab_ts

    def make_index_list(self, mixed_indices):  # Extracts masked and unmasked data indices from randomly shuffled mixed indices list
        mixed_indices.sort()
        chars, nums = zip(*mixed_indices)
        
        char_c = Counter(chars)
        return list(nums[:char_c['m']]), list(nums[char_c['m']:])

    def get_cross_entropy_class_weights(self):  # Generates weights for classes for CrossEntropyLoss (Weights to use for imbalanced datasets)
        x_img_mask_tr, _, x_img_umsk_tr, _ = self.fetch_train_data()
        size_list  = [x_img_umsk_tr.shape[0], x_img_mask_tr.shape[0]]    # Getting data size
        
        # Weight calculation and returning them
        return torch.tensor([1 - (class_size / sum(size_list)) for class_size in size_list])

    def get_train_data(self, epoch=10):            # Fetching train data to the model
        x_img_mask_tr, y_lab_mask_tr, x_img_umsk_tr, y_lab_umsk_tr = self.fetch_train_data()
        mask_tr_set_len, umsk_tr_set_len  = x_img_mask_tr.shape[0], x_img_umsk_tr.shape[0]  # Dataset length for masked and unmasked training images
        train_set_len = umsk_tr_set_len + mask_tr_set_len       # Total training set length

        assert self.batch_size <= train_set_len, "Error: Batch size is greater than total train data available"
    
        num_batches = int(train_set_len/self.batch_size)    # Number of batches
        index_list = [('m', i) for i in range(mask_tr_set_len)] + [('u', i) for i in range(umsk_tr_set_len)]    # List to combine indices of two datasets

        for i in range(epoch):
            batch, offset = 0, self.batch_size
            start, end = 0, 0
            num_list_temp = list(range(self.batch_size))

            # Randomly shuffling data indices at each epoch
            random.shuffle(index_list)

            while batch < num_batches:
                if (train_set_len - (end + 2*offset)) < 0:      # Mechanism to train all the batch data even when train_sel_len is not completely divisible by self.batch_size
                    offset = offset + train_set_len%offset

                start, end, batch = end, end+offset, batch+1
                
                temp_idx = index_list[start:end]
                temp_idx_li_m, temp_idx_li_u = self.make_index_list(temp_idx)   # Getting indices for Masked and Unmasked datasets

                '''
                In HDF5, the indexing must be in sorted order, therefore we are randomly shuffling the index list and 
                sorting each batch of indices above to acces data from HDF5 dataset.
                '''

                # Variables to store data
                X_m, Y_m, X_u, Y_u = None, None, None, None

                if temp_idx_li_m:
                    # Extracting Masked Images
                    X_m = torch.tensor(x_img_mask_tr[temp_idx_li_m], dtype=torch.float32).permute(0, 3, 1, 2)   # NCHW format
                    Y_m = torch.tensor(y_lab_mask_tr[temp_idx_li_m], dtype=torch.long)

                if temp_idx_li_u:
                    # Extracting Unmasked Images
                    X_u = torch.tensor(x_img_umsk_tr[temp_idx_li_u], dtype=torch.float32).permute(0, 3, 1, 2)   # NCHW format
                    Y_u = torch.tensor(y_lab_umsk_tr[temp_idx_li_u], dtype=torch.long)

                if X_m == None:
                    X_comb, Y_comb = X_u, Y_u
                elif X_u == None:
                    X_comb, Y_comb = X_m, Y_m
                else:
                    # Concatenating along zeroth dimension
                    X_comb = torch.cat((X_m, X_u), 0)
                    Y_comb = torch.cat((Y_m, Y_u), 0)

                # Again randomly shuffling the array
                if X_comb.shape[0] < self.batch_size:
                    num_list_temp = list(range(X_comb.shape[0]))
                random.shuffle(num_list_temp)

                X_comb = X_comb[num_list_temp]
                Y_comb = Y_comb[num_list_temp]

                yield i+1, batch, num_batches, X_comb, Y_comb    # returns epoch_number, batch_number, total_batches, image_array, labels_array (Dimenson-> X: (batch_size, 3, H, W), Y: (batch_size, num_class))

    def get_test_data(self):                       # Fetching test data to the model
        if self.test_X == None or self.test_Y == None:
            x_img_ts, y_lab_ts = self.fetch_test_data()
            test_set_len = x_img_ts.shape[0]
            
            num_list = list(range(test_set_len))

            self.test_X = torch.tensor(x_img_ts[num_list], dtype=torch.float32).permute(0, 3, 1, 2)     # NCHW format
            self.test_Y = torch.tensor(y_lab_ts[num_list], dtype=torch.long)

        yield self.test_X, self.test_Y      # Dimenson-> test_X: (N, 3, H, W), test_Y: (N, num_class)

if __name__ == '__main__':
    data_driver = object_data('../data/self-built-masked-face-recognition-dataset/')

    for epoch, batch, total_batch, X, Y in data_driver.get_train_data():
        print(X.shape, Y.shape, total_batch)
        print(Y)
        print(X)

        # plt.imshow(X[0].permute(1, 2, 0).data.numpy())
        # plt.show()
        break

    for X, Y in data_driver.get_test_data():
        print(X.shape, Y.shape)
        print(Y)
        print(X)
        break
