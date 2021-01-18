import torch
import numpy as np
import collections

#adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py

class iouEval:
    def __init__(self, nClasses):
        self.nClasses = nClasses
        self.reset()

    def reset(self):
        self.overall_acc = 0
        self.per_class_acc = np.zeros(self.nClasses, dtype=np.float32)
        self.per_class_iu = np.zeros(self.nClasses, dtype=np.float32)
        self.mIOU = 0
        self.batchCount = 1
        self.count = collections.Counter()

    def fast_hist(self, a, b):
        k = (a >= 0) & (a < self.nClasses)
        return np.bincount(self.nClasses * a[k].astype(int) + b[k], minlength=self.nClasses ** 2).reshape(self.nClasses, self.nClasses)

    def compute_hist(self, predict, gth):
        hist = self.fast_hist(gth, predict)
        return hist

    def addBatch(self, predict, gth):
        for predict,gth in zip(predict,gth):
            predict = predict.to("cpu").numpy().flatten()
            gth = gth.to("cpu").numpy().flatten()
            epsilon = 0.00000001
            hist = self.compute_hist(predict, gth)
            overall_acc = np.diag(hist)[1:].sum() / (hist[1:][:,1:].sum() + epsilon)
            per_class_acc = np.diag(hist) / (hist.sum(1) + epsilon)
            per_class_iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)
            l = np.unique(gth)
            mask = np.bincount(l, minlength=self.nClasses)
            mask[0] = 0
            #l = set([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33])
            #mask = np.bincount(np.array(list(l&set(np.unique(gth)))), minlength=self.nClasses)
            mIou = np.nanmean(per_class_iu[mask!=0])
            #print(mIou)

            self.overall_acc +=overall_acc
            self.per_class_acc += per_class_acc
            self.per_class_iu += per_class_iu
            self.mIOU += mIou
            self.batchCount += 1
            self.count += collections.Counter(l)

    def getMetric(self):
        overall_acc = self.overall_acc / self.batchCount
        per_class_acc = self.per_class_acc
        per_class_iu = self.per_class_iu
        mIOU = self.mIOU / self.batchCount
        for i in self.count:
            if i==0:continue
            per_class_acc[i] /= self.count[i]
            per_class_iu[i] /= self.count[i]         
        #overall_acc = np.nanmean(per_class_acc)
        #mIOU = np.nanmean(per_class_iu)
        return overall_acc, per_class_acc, per_class_iu, mIOU