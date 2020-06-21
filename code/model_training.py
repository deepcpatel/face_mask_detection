import os
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from model import mask_detector
from data_loader import object_data
from sklearn.metrics import f1_score, accuracy_score

class model_training():
    def __init__(self):
        super(model_training, self).__init__()

        self.name = "Mask_Detection"
        self.data_dir = "../data/self-built-masked-face-recognition-dataset/"
        self.batch_size = 100                   # Num of batches
        self.ml_model = None                    # Stores Classifier model. Will be initialized later
        self.epoch = 5                          # Number of training epochs
        self.resize_size = (100, 100)           # Data Resize Size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # ADAM Parameters
        self.adam_lr = 1e-4
        self.adam_betas = (0.9, 0.999)
        self.adam_eps = 1e-8
        self.optimizer = None   # Will store Optimizer

        # Invoking Object data loader object
        self.obj_data = object_data(data_dir=self.data_dir, batch_size=self.batch_size, img_resize_size=self.resize_size)
        self.B_Loss = nn.CrossEntropyLoss(weight=self.obj_data.get_cross_entropy_class_weights())    # Cross Entropy loss

    def get_task_name(self):
        return self.name

    def initiaize_model(self):
        self.ml_model = mask_detector().to(self.device)

    def init_optimizer(self):
        # Initializing optimizer
        self.optimizer = optim.Adam(self.ml_model.parameters(), lr = self.adam_lr, betas = self.adam_betas, eps = self.adam_eps)

    def calc_loss(self, Y_pred, Y):
        # Y_pred -> dim: (batch_size, 1)
        # Y      -> dim: (batch_size, 1)
        # Labels -> 0: Unmasked, 1: Masked
        loss = self.B_Loss(Y_pred, Y)
        return loss
    
    def save_ml_model(self, epoch):                 # Saves trained classifier model
        if not os.path.exists("saved_models/" + self.name):
            os.mkdir("saved_models/" + self.name)
        state_dic = {'task_name': self.name, 'state_dict': self.ml_model.state_dict()}
        filename = "saved_models/" + self.name + "/mask_predictor_" + str(epoch) + ".pth.tar"
        torch.save(state_dic, filename)

    def load_ml_model(self, epoch, option=1):       # Loads trained classifier model
        path = "saved_models/" + self.name + "/mask_predictor_" + str(epoch) + ".pth.tar"
        if option == 1:             # Loading for training
            checkpoint = torch.load(path, map_location=self.device)
            self.ml_model.load_state_dict(checkpoint['state_dict'])
        else:                       # Loading for testing
            checkpoint = torch.load(path, map_location=self.device)
            self.ml_model.load_state_dict(checkpoint['state_dict'])
            self.ml_model.eval()
        
    def calc_cost(self, Y_true, Y_pred):            # Calculates cost metric
        accuracy = accuracy_score(Y_true, Y_pred)   # Accuracy
        f1 = f1_score(Y_true, Y_pred)               # F1 Score
        return accuracy, f1

    def train(self):
        loss_list = []  # Stores loss for all epoch and iteration
        prev_epoch = 1

        # Loading pretrained ML model
        # self.load_ml_model(epoch=2)

        for epoch_no, batch_no, total_batch, X, Y in self.obj_data.get_train_data(epoch=self.epoch):
            
            if epoch_no>prev_epoch:     # Testing before starting new epoch
                self.save_ml_model(prev_epoch)
                self.test(prev_epoch)
                prev_epoch = epoch_no

            X, Y = X.to(self.device), Y.view(-1).to(self.device)   # Sending data to appropriate device
            
            Y_pred = self.ml_model(X)  # Output dimension : (batch_size x 2)

            loss = self.calc_loss(Y_pred, Y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())
            print("Epoch: " + str(epoch_no) + "/" + str(self.epoch) + ", Batch: " + str(batch_no) + "/" + str(total_batch) + ", Loss: " + str(loss_list[-1]))

        self.save_ml_model(prev_epoch)
        self.test(prev_epoch)
        return loss_list
    
    def test(self, epoch, load='N'):
        if load == 'Y':
            self.load_ml_model(epoch, option=2)     # Loading Prediction model trained for given epoch
        else:
            self.ml_model.eval()    # Changing model to testing mode

        for X, Y in self.obj_data.get_test_data():
            X = X.to(self.device)            # Sending data to appropriate device
            Y_pred = torch.argmax(F.softmax(self.ml_model(X), dim=1), dim=1) # Output dimension : (batch_size x 1)

            # Converting to NumPy array
            Y = Y.view(-1).cpu().data.numpy()
            Y_pred = (Y_pred.view(-1).cpu().data.numpy()).astype(np.int32)
            
            accuracy, f1 = self.calc_cost(Y, Y_pred)
            print("\nEpoch " + str(epoch) + " completed! Accuracy: " + str(accuracy) + ", Micro F1 score: " + str(f1) + "\n")

        self.ml_model.train()   # Making model trainable again

if __name__ == '__main__':
    mt = model_training()   # Generates Feature

    mt.initiaize_model()    # Initializes model
    mt.init_optimizer()     # Initializes Optimizer

    loss_list = mt.train()  # Training
    
    '''
    mt.test(epoch = 1, load='Y')          # Testing
    mt.test(epoch = 3, load='Y')          # Testing
    mt.test(epoch = 5, load='Y')          # Testing
    '''