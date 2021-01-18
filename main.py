import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import torch.nn as nn
import math
import numpy as np
import torch.backends.cudnn as cudnn

import loadData as ld
import Transforms as myTransforms
import DataSet as myDataLoader
import drn

import pickle
import numpy as np
import argparse
import random
import os

import model

from solver import Solver
import arguments
import time


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return self.softmax(y), x

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param


def cifar_transformer():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5,], std=[0.5, 0.5, 0.5]),
        ]
    )


def main(args):
    # data load
    # city scapes -> train:2975 val:500 test:1525
    if not os.path.isfile(args.cached_data_file) or args.data_load:
        dataLoad = ld.LoadData(args.data_dir, args.classes, args.cached_data_file)
        dataLoad = dataLoad.processData()
        if dataLoad is None:
            print("Error while pickling data. Please check.")
            exit(-1)
    else:
        dataLoad = pickle.load(open(args.cached_data_file, "rb"))

    args.initial_budget = args.num_images//10
    args.budget = args.num_images//10
    all_indices = np.arange(args.num_images)
    initial_indices = random.sample(list(all_indices), args.initial_budget)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    weight = torch.from_numpy(dataLoad["classWeights"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight = weight.to(device)
    info = {"std": [0.1829540508368939, 0.18656561047509476, 0.18447508988480435], "mean": [0.29010095242892997, 0.32808144844279574, 0.28696394422942517]}

    trainDataset_main = myTransforms.Compose(
        [
            myTransforms.Normalize(mean=info["mean"], std=info["std"]),
            myTransforms.Scale(688, 688),
            #myTransforms.RandomCropResize(32),
            myTransforms.RandomFlip(),
            myTransforms.RandomRotate(),
            # myTransforms.RandomCrop(64).
            myTransforms.ToTensor(args.scaleIn),
            #
        ]
    )

    valDataset = myTransforms.Compose(
        [
            myTransforms.Normalize(mean=info["mean"], std=info["std"]),
            myTransforms.Scale(688, 688),
            myTransforms.ToTensor(args.scaleIn),
            #
        ]
    )

    val_dataset, test_dataset = torch.utils.data.random_split(myDataLoader.MyDataset(dataLoad["valIm"], dataLoad["valAnnot"], transform=valDataset), [400, 100])

    # dataset with labels available
    querry_dataloader = data.DataLoader(
        myDataLoader.MyDataset(
            dataLoad["trainIm"], dataLoad["trainAnnot"], transform=trainDataset_main
        ),
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False
    )

    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    cudnn.benchmark = True

    args.cuda = args.cuda and torch.cuda.is_available()
    solver = Solver(args, test_dataloader, weight)

    splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    current_indices = list(initial_indices)

    accuracies = []
    ious = []

    for split in splits:
        single_model = DRNSeg(args.arch, args.classes, None, pretrained=True)
        task_model = torch.nn.DataParallel(single_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        task_model = task_model.to(device)
        vae = model.VAE(args.latent_dim)
        discriminator = model.Discriminator(args.latent_dim)

        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(
            myDataLoader.MyDataset(
                dataLoad["trainIm"], dataLoad["trainAnnot"], transform=trainDataset_main
            ),
            sampler=unlabeled_sampler,
            batch_size=args.batch_size,
            drop_last=False,
        )

        # train the models on the current data
        final_mIOU, overall_acc, task_model, vae, discriminator = solver.train(
            querry_dataloader, val_dataloader, task_model, single_model, vae, discriminator, unlabeled_dataloader
        )

        print("Final mIOU with {}% of data is: {:.2f}".format(int(split * 100), final_mIOU))
        print("Final acc with {}% of data is: {:.2f}".format(int(split * 100), overall_acc))
        accuracies.append(overall_acc)
        ious.append(final_mIOU)
        print(accuracies)
        print(ious)

        sampled_indices = solver.sample_for_labeling(vae, discriminator, unlabeled_dataloader)
        current_indices = list(current_indices) + list(sampled_indices)
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = data.DataLoader(
            myDataLoader.MyDataset(dataLoad["trainIm"], dataLoad["trainAnnot"], transform=trainDataset_main),
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False
            )

    print(accuracies)
    print(ious)



if __name__ == "__main__":
    ut = time.time()
    args = arguments.get_args()
    SEED = 114
    fix_seed(SEED)
    main(args)
    print(time.time()-ut)
