import torch
import cv2
import torch.utils.data

__author__ = "Sachin Mehta"


class MyDataset(torch.utils.data.Dataset):
    """
    Class to load the dataset
    """

    def __init__(self, imList, labelList, transform=None):
        """
        :param imList: image list (Note that these lists have been processed and pickled using the loadData.py)
        :param labelList: label list (Note that these lists have been processed and pickled using the loadData.py)
        :param transform: Type of transformation. SEe Transforms.py for supported transformations
        """
        self.imList = imList
        self.labelList = labelList
        self.transform = transform

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        """

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        """
        image_name = self.imList[idx]
        label_name = self.labelList[idx]
        image = cv2.imread(image_name)
        label = cv2.imread(label_name, 0)
        mask = [0, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        mask_zero = [1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34]
        for i in mask_zero:
            label[label == i] = 0
        for i, j in enumerate(mask):
            label[label == j] = i
        if self.transform:
            [image, label] = self.transform(image, label)
        return (image, label, idx)
