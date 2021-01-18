import torch
import torch.nn as nn
import numpy as np


class AdversarySampler:
    def __init__(self, budget, args):
        self.budget = budget
        self.args = args

    def sample(self, vae, discriminator, data, cuda):
        all_preds = []
        all_indices = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for images, _, indices in data:
            images = images.to(device)

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices
