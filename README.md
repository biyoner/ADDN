# ADDN
Adversarial and densely dilated network

## Introduction 
In this work, we propose densely and dilated network to apply connectomes segmentation problem. 

## ADDN architecture
<img src="https://github.com/biyoner/ADDN/blob/master/results/f2.png" width="500" height="500" alt="图片加载失败时，显示这段字"/>

## Experimental results
We perform our experiment on two datasets to evaluate the effectiveness of the ADDN model.
<img src="https://github.com/biyoner/ADDN/blob/master/results/f1.png" width="600" height="350" alt="图片加载失败时，显示这段字"/>

## Configure the network
All network hyperparameters are configured in ADDN.py.

### Training
epochs: how many iterations or steps to train  
sample_interval: how many steps to perform a visulization  

### Data
data_path: data path  
img_trn, msk_trn: .npy file for training data  
img_val, msk_val: .npy file for validation  
img_tst: .npy file for testing   
input_shape, output_shape: height, width and channl of the input image and output image    
patch: PatchGAN output   

## Training and Testing 
### Start training:
After configure the network, we can start to train. Run
```
python ADDN train \  
       --train
```
The training of ADDN for semantic segmentation will start.

### Training process visualization
We employ tensorboard to visualize the training process.
```
tensorboard --logdir=Graph/
```
The segmentation results including training and validation losses are available in tensorboard.

### Validation
Choose the validation mode:
```
python ADDN.py \
       --val \  
       --weight checkpoints/ADDN/weights_loss_trn.weights
```

### Testing and prediction
Choose the test mode:
```
python ADDN.py \
       --submit \  
       --weight checkpoints/ADDN/weights_loss_trn.weights
```

## Use ADDN
1. Data preparation  
2. Change the generator and discriminator network to your own ones
