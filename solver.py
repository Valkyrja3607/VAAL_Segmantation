import torch
import torch.nn as nn
import torch.optim as optim

#from Criteria import CrossEntropyLoss2d
from IOUEval import iouEval

import os
import numpy as np
from sklearn.metrics import accuracy_score

import sampler
import copy


class Solver:
    def __init__(self, args, test_dataloader, weight):
        self.args = args
        self.test_dataloader = test_dataloader

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.criterion = nn.NLLLoss2d()#ignore_index=0)

        self.sampler = sampler.AdversarySampler(self.args.budget, self.args)

    def adjust_learning_rate(self, optimizer, epoch):
        """
        Sets the learning rate to the initial LR decayed by 10 every 30 epochs
        """
        if self.args.lr_mode == 'step':
            lr = self.args.lr * (0.1 ** (epoch // self.args.lr_step))
        elif self.args.lr_mode == 'poly':
            lr = self.args.lr * (1 - epoch / self.args.train_epochs) ** 0.9
        else:
            raise ValueError('Unknown lr mode {}'.format(self.args.lr_mode))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img

    def train(self, querry_dataloader, val_dataloader, task_model, single_model, vae, discriminator, unlabeled_dataloader):
        self.args.train_iterations = (self.args.num_images * self.args.train_epochs) // self.args.batch_size
        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)

        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_task_model = optim.Adam(single_model.optim_parameters(),
                                self.args.lr,
                                weight_decay=self.args.weight_decay)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optim_task_model, step_size=args.step_loss, gamma=0.5)

        vae.train()
        discriminator.train()
        task_model.train()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        task_model = task_model.to(device)
        self.criterion = self.criterion.to(device)
        vae = vae.to(device)
        discriminator = discriminator.to(device)

        best_IOU = 0
        best_acc = 0
        for i in range(self.args.train_epochs):
            for labeled_imgs, labels, _ in querry_dataloader:
                labeled_imgs = labeled_imgs.to(device)
                labels = labels.to(device)
                
                preds = task_model(labeled_imgs)[0]
                task_loss = self.criterion(preds, labels)
                optim_task_model.zero_grad()
                task_loss.backward()
                optim_task_model.step()

            print("Current training epochs: {}".format(i))
            print("Current task model loss: {:.4f}".format(task_loss.item()))

            overall_acc, per_class_acc, per_class_iu, mIOU = self.validate(task_model, val_dataloader)
            if mIOU > best_IOU:
                best_IOU = mIOU
                best_model = copy.deepcopy(task_model)
            best_acc = max(best_acc, overall_acc)

            print("current step: {} mIOU: {}".format(i, mIOU))
            print("all acc:", overall_acc)
            print("best IOU: ", best_IOU)
        for i in range(self.args.train_iterations):
            self.adjust_learning_rate(optim_task_model, i*self.args.batch_size//self.args.num_images)
            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)
            # task_model step
            labeled_imgs = labeled_imgs.to(device)
            labels = labels.to(device)
            unlabeled_imgs = unlabeled_imgs.to(device)
            
            # VAE step
            recon, z, mu, logvar = vae(labeled_imgs)
            unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.args.beta)

            unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
            transductive_loss = self.vae_loss(unlabeled_imgs, unlab_recon, unlab_mu, unlab_logvar, self.args.beta)
        
            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)

            lab_real_preds = torch.ones(labeled_imgs.size(0)).view(-1,1)
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0)).view(-1,1)
                    
            lab_real_preds = lab_real_preds.to(device)
            unlab_real_preds = unlab_real_preds.to(device)

            dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + self.bce_loss(unlabeled_preds, unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + self.args.adversary_param * dsc_loss
            optim_vae.zero_grad()
            total_vae_loss.backward()
            optim_vae.step()

            # Discriminator step
            with torch.no_grad():
                _, _, mu, _ = vae(labeled_imgs)
                _, _, unlab_mu, _ = vae(unlabeled_imgs)

            labeled_preds = discriminator(mu)
            unlabeled_preds = discriminator(unlab_mu)

            lab_real_preds = torch.ones(labeled_imgs.size(0)).view(-1,1)
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0)).view(-1,1)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            lab_real_preds = lab_real_preds.to(device)
            unlab_fake_preds = unlab_fake_preds.to(device)

            dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + self.bce_loss(unlabeled_preds, unlab_fake_preds)

            optim_discriminator.zero_grad()
            dsc_loss.backward()
            optim_discriminator.step()

            if i % 100 == 0:
                print("Current training epochs: {}".format(i))
                print("Current task model loss: {:.4f}".format(task_loss.item()))

                overall_acc, per_class_acc, per_class_iu, mIOU = self.validate(task_model, val_dataloader)
                if mIOU > best_IOU:
                    best_IOU = mIOU
                    best_model = copy.deepcopy(task_model)
                best_acc = max(best_acc, overall_acc)

                print("current step: {} mIOU: {}".format(i, mIOU))
                print("all acc:", overall_acc)
                print("best IOU: ", best_IOU)

        best_model = best_model.to(device)

        overall_acc, per_class_acc, per_class_iu, final_mIOU = self.test(best_model)
        return final_mIOU, overall_acc, task_model, vae, discriminator

    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader):
        querry_indices = self.sampler.sample(
            vae, discriminator, unlabeled_dataloader, self.args.cuda
        )

        return querry_indices

    def validate(self, task_model, loader):
        task_model.eval()
        iouEvalVal = iouEval(self.args.classes)

        total, correct = 0, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = task_model(imgs)[0]

            iouEvalVal.addBatch(preds.max(1)[1].detach(), labels.detach())
            overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalVal.getMetric()

        return overall_acc, per_class_acc, per_class_iu, mIOU

    def test(self, task_model):
        task_model.eval()
        iouEvalTest = iouEval(self.args.classes)

        total, correct = 0, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for imgs, labels, _ in self.test_dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = task_model(imgs)[0]

            iouEvalTest.addBatch(preds.max(1)[1].detach(), labels.detach())
            overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalTest.getMetric()

        return overall_acc, per_class_acc, per_class_iu, mIOU

    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD
