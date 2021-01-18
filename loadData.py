import numpy as np
import cv2
import pickle
import torch
#from torch.autograd import Variable
import glob
from PIL import Image as PILImage
#import Model as Net
import os
import time
from argparse import ArgumentParser


class LoadData:
    """
    Class to laod the data
    """

    def __init__(self, data_dir, classes, cached_data_file, normVal=1.10):
        """
        :param data_dir: directory where the dataset is kept
        :param classes: number of classes in the dataset
        :param cached_data_file: location where cached file has to be stored
        :param normVal: normalization value, as defined in ERFNet paper
        """
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.trainImList = list()
        self.valImList = list()
        self.testImList = list()
        self.trainAnnotList = list()
        self.valAnnotList = list()
        self.testAnnotList = list()
        self.cached_data_file = cached_data_file

    def compute_class_weights(self, histogram):
        """
        Helper function to compute the class weights
        :param histogram: distribution of class samples
        :return: None, but updates the classWeights variable
        """
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readFile(self, fileName, trainStg=False):
        """
        Function to read the data
        :param fileName: file that stores the image locations
        :param trainStg: if processing training or validation data
        :return: 0 if successful
        """
        if trainStg == True:
            global_hist = np.zeros(self.classes, dtype=np.float32)

        no_files = 0
        with open(self.data_dir + "/" + fileName, "r") as textFile:
            for line in textFile:
                # we expect the text file to contain the data in following format
                # <RGB Image>, <Label Image>
                line_arr = line.split(",")
                img_file = ((self.data_dir).strip() + line_arr[0].strip()).strip()
                label_file = ((self.data_dir).strip() + line_arr[1].strip()).strip()
                label_file = label_file.replace("labelTrainIds", "labelIds")

                if trainStg == True:
                    self.trainImList.append(img_file)
                    self.trainAnnotList.append(label_file)
                else:
                    self.valImList.append(img_file)
                    self.valAnnotList.append(label_file)

                no_files += 1

        if trainStg == True:
            # divide the mean and std values by the sample space size
            self.mean /= no_files
            self.std /= no_files

            # compute the class imbalance information
            self.compute_class_weights(global_hist)
        return 0
    
    def readTest(self,path_list):
        for img_path in path_list:
            label_path=img_path.replace("leftImg8bit.png","gtFine_labelIds.png").replace("leftImg8bit","gtFine")
            #label_img = cv2.imread(label_path, 0)
            self.testImList.append(img_path)
            self.testAnnotList.append(label_path)
        

    def processData(self):
        """
        main.py calls this function
        We expect train.txt and val.txt files to be inside the data directory.
        :return:
        """
        print("Processing training data")
        return_val = self.readFile("train.txt", True)

        print("Processing validation data")
        return_val1 = self.readFile("val.txt")

        print("Processing test data")
        test_list=[]
        for j in ["berlin","bielefeld","bonn","leverkusen","mainz","munich"]:
            path=self.data_dir + "/leftImg8bit" + "/test/" + j
            test_list.extend(glob.glob(path + '/*.png'))
        return_val2 = self.readTest(test_list)

        print("Pickling data")
        if return_val == 0 and return_val1 == 0:
            data_dict = dict()
            data_dict["trainIm"] = self.trainImList
            data_dict["trainAnnot"] = self.trainAnnotList
            data_dict["valIm"] = self.valImList
            data_dict["valAnnot"] = self.valAnnotList
            data_dict["testIm"] = self.testImList
            data_dict["testAnnot"] = self.testAnnotList
            data_dict["classWeights"] = self.classWeights

            pickle.dump(data_dict, open(self.cached_data_file, "wb"))
            return data_dict
        return None

