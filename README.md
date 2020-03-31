# Gesture classification through videos using hybrid CNN - RNN networks and Transfer learning.

Link to the dataset:
- [Dataset](https://drive.google.com/open?id=1KdyHxZxQlvZGgPlAHG-eJXSuZQ3S9Kns)

Links to the trained models can be found below:
- [InceptionV3 with GRU](https://drive.google.com/open?id=1Z7Jd-DGfkLr9ZOkNOqbjKBGXD0k_uOWn) - 93% validation accuracy - 260Mb
- [MobileNets with GRU](https://drive.google.com/open?id=1MPBCbcfBKjDg2-IC2LdVLO6G0JEHEX4v) - 89% validation accuracy - 42Mb

Training for all networks have been performed using Nvidia Tesla K80. For more details visit, [Nimblebox.](https://www.nimblebox.ai/)

#### Experiment 1: Conv2D+LSTM 
Six Conv2D (Batch normalization+ Max Pooling) Layer -> LSTM -> Dense Layer -> Softmax

Batch size:  32 Epoch: 25 Training Samples: 256 Total Parameters: 4,010,725

Result: Error: MemoryResources: GPU is unable to load entire batch into memory

#### Experiment 2: Conv2D+LSTM 
Six Conv2D (Batch normalization+ Max Pooling) Layer -> LSTM -> Dense Layer -> Softmax

Batch size:  16 Epoch: 15 Training Samples: FULL Total Parameters: 4,010,725

Result: Validation Accuracy = 34%

Decision: As observed in the curves the sudden increase in validation loss could be due to optimization oscillating around a minima. Adam uses 0.01 as default learning rate, we decided to experiment with lower learning rates.
 
#### Experiment 3: Conv2D+LSTM 
Six Conv2D (Batch normalization+ Max Pooling) Layer -> LSTM -> Dense Layer -> Softmax

Batch size:  16 Epoch: 15 Training Samples: FULL Total Parameters: 4,010,725

Result: Validation Accuracy = 46%

Decision: learning rate 0.002 gave best results with faster training as compared to lower rates and more subdued spikes. The accuracy is still a concern so we decided to use some data augmentation.

#### Experiment 4: Conv2D+LSTM 
Six Conv2D (Batch normalization+ Max Pooling) Layer -> LSTM -> Dense Layer -> Softmax

Batch size:  16 Epoch: 15 Training Samples: FULL Total Parameters: 4,010,725

Result: Validation Accuracy = 69.9% with Gaussian noise technique for data augmentation

Decision: We tried variety of data augmentation techniques such as affine transformations and salt & pepper noise but all the processes tremendously slowed down training. So we finally decided to go with adding Gaussian noise  

#### Experiment 5: Conv2D+GRU
Six Conv2D (Batch normalization+ Max Pooling) Layer -> GRU -> Dense Layer -> Softmax

Batch size:  16 Epoch: 15 Training Samples: FULL Total Parameters: 1,831,525

Result: Validation Accuracy = 70.87% with Gaussian noise technique for data augmentation

Decision: GRU gave similar validation accuracy with fewer parameters. 
 
#### Experiment 6: Conv2D+GRU
Five Conv2D (Batch normalization+ Max Pooling) Layer -> GRU -> Dense Layer -> Softmax

Batch size:  16 Epoch: 10 Training Samples: FULL Total Parameters: 853,925

Result: Validation Accuracy = 54.6% with Gaussian noise technique for data augmentation

Decision: Removing one Conv2D layer had a steep impact on the performance. 
 
#### Experiment 7: Conv2D+GRU
Five Conv2D (Batch normalization+ Max Pooling) Layer -> GRU -> Dense Layer -> Softmax

Batch size:  16 Epoch: 20 Training Samples: FULL Total Parameters: 2,560,677

Result: Validation Accuracy = 77% with Gaussian noise technique for data augmentation

Decision: Adding back the Conv2D layer, increasing number of GRU cells and increasing number of epochs. 
 
#### Experiment – 8: Transfer learning
Pre-trained InceptionV3 Layer (Non-trainable) -> GRU -> Dense Layer -> Softmax

Batch size:  16 Epoch: 20 Training Samples: FULL Total Parameters: 22,655,525

Result: Validation Accuracy = 40% with Gaussian noise technique for data augmentation

Decision: Picking up InceptionV3 with pre-trained weights for ImageNet (For it’s smaller number of parameters and state of the art performance) and training only the GRU layer doesn’t result in good results. 
 
#### Experiment – 9: Transfer learning
InceptionV3 Layer (Trainable) -> GRU -> Dense Layer -> Softmax

Batch size:  16 Epoch: 16 Training Samples: FULL Total Parameters: 22,655,525

Result: Validation Accuracy = 93% with Gaussian noise technique for data augmentation

Decision: Allowing inception layer to train itself results in substantial improvement in accuracy.
 
#### Experiment – 10: Transfer learning
MobileNets Layer (Trainable) -> GRU -> Dense Layer -> Softmax

Batch size:  16 Epoch: 16 Training Samples: FULL Total Parameters: 3,684,293

Result: Validation Accuracy = 89% with Gaussian noise technique for data augmentation

Decision: Chose MobileNets for its smaller size with only 1/7th the total number of parameters, we are able to achieve comparable accuracies to the model with InceptionV3.
 
#### Conclusion:
For tasks requiring higher accuracy and no system constraints, clearly, the model with InceptionV3 as the feature extraction unit will perform better and given more time to train can achieve even higher accuracies. 

For embedded systems workload with system constraint, model with MobileNet as the feature extraction unit provides a wonderful alternative. 

