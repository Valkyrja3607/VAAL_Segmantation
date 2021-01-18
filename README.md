# VAAL SemanticSegmantation in PyTorch
Reproduction implementation of VAAL([Variational Adversarial Active Learning](https://arxiv.org/abs/1904.00370)) Segmantation Task.
The classification task has an author implementation [here](https://github.com/sinhasam/vaal)

### Prerequisites:
- Linux or macOS
- Python 3.8
- CPU compatible but NVIDIA GPU + CUDA CuDNN is highly recommended.

### Experiments and Visualization
Please edit --data_dir in auguments.py to where your Cityscapes data is.


The code can simply be run using 
```
python3 main.py
```
If you want to use GPU
```
python3 main.py --cuda
```
When using the model with different datasets or different variants, the main hyperparameters to tune are
```
--adversary_param --beta --num_vae_steps and --num_adv_steps
```

The results will be saved in `results/accuracies.log`. 
