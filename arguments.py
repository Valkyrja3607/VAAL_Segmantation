import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
    parser.add_argument('--dataset', type=str, default='cityscapes', help='Name of the dataset used.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size used for training and testing')
    parser.add_argument('--train_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--latent_dim', type=int, default=32, help='The dimensionality of the VAE latent dimension')
    parser.add_argument('--beta', type=float, default=1, help='Hyperparameter for training. The parameter for VAE')
    parser.add_argument('--num_images', type=int, default=2475, help='Batch size used for training and testing')
    parser.add_argument('--num_adv_steps', type=int, default=1, help='Number of adversary steps taken for every task model step')
    parser.add_argument('--num_vae_steps', type=int, default=2, help='Number of VAE steps taken for every task model step')
    parser.add_argument('--adversary_param', type=float, default=1, help='Hyperparameter for training. lambda2 in the paper')
    parser.add_argument('--out_path', type=str, default='./results', help='Path to where the output log will be')
    parser.add_argument('--log_name', type=str, default='accuracies.log', help='Final performance of the models will be saved with this name')
    parser.add_argument("--cached_data_file", default="city.p", help="Cached file name")
    parser.add_argument("--data_load", action='store_true', help="load data")
    parser.add_argument("--data_dir", default="./city", help="Data directory")
    parser.add_argument("--classes", type=int, default=20, help="No of classes in the dataset. 20 for cityscapes",)
    parser.add_argument("--savedir", default="./results", help="directory to save the results")
    parser.add_argument("--scaleIn", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1, help="No. of parallel threads")
    parser.add_argument('--arch', type=str, default="drn_d_22")
    parser.add_argument('--lr_mode', type=str, default='poly')
    parser.add_argument("--lr", type=float, default=0.0003, help="Initial learning rate")#0.0005
    parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    
    return args
