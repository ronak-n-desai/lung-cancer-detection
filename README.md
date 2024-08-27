# Lung Cancer Detection with Histopathological Images

**Authors: Abuduaini Niyazi, Derek Kielty, Rishat Dilmurat, Sujoy Upadhyay, Ronak Desai**

*Submitted as a final project for the [Erdos Institute](https://www.erdosinstitute.org/) Deep Learning (Summer 2024) Boot Camp*

## Introduction

One of the most important applications of deep neural networks in the healthcare industry is the classification of various types of diseases from image data. Particularly, one life-saving application is to detect cancerous cells at an early stage. In this project, we will train convolutional neural networks (CNNs) with a [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) of 15,000 images to distinguish normal lung tissues from cancerous ones.

![image](https://github.com/user-attachments/assets/c89b7429-ab99-4678-9613-5fa1f08e6a53)

The above images are (from left to right) adenocarcinoma (ACA), squamous cell carcinoma (SCC), and normal lung tissue (N). They are samples of images found in the data set. Not only do we need to differentiate cancer from no cancer, but we need to differentiate between the two types of lung cancer.

[Some others](https://www.geeksforgeeks.org/lung-cancer-detection-using-convolutional-neural-network-cnn/#) have done an analysis on this with varying levels of accuracy. The [analysis with the most upvotes (78 upvotes with a gold medal)](https://www.kaggle.com/code/mohamedsameh0410/lung-cancer-detection-with-cnn-efficientnetb3#EfficientNetB3) used [TensorFlow](https://www.tensorflow.org/) to train a CNN model with 94.67% accuracy on a validation set. Furthermore, using a pre-trained [EfficientNetB3](https://keras.io/api/applications/efficientnet/) model (pre-trained with [ImageNet](https://image-net.org/) data), they achieved 100 percent accuracy on the validation set.

In this project, our group aims to match or exceed the validation set accuracy of the other groups that have worked with this dataset previously. Additionally, we use the frameworm of [PyTorch](https://pytorch.org/) to perform our machine learning analysis. 

## Pre-processing

The dataset that we are using has a total of 15,000 RGB images of size 768 by 768 pixels. The images are already evenly split into 5,000 images each of ACA, SCC, and N. Each of the 5,000 images was originally composed of only 250 images and through [data augmentation](https://pytorch.org/vision/stable/transforms.html) (rotations, shearing, mirroring, etc.) a more robust set of 5,000 was obtained. To standardize the comparisons, we scale the image size down to 256 by 256, use a batch size of 64, and use a training-validation split of 80-20. 

## Methods

In carrying out this project, we studied CNN architectures and deep learning techniques that have been used successfully in the past. We started by constructing a basic architecture composed of a few convolutional blocks with max pooling followed by a few dense (fully connected) layers to the three outputs. We experimented with various techniques to improve the model accuracy such as increasing the width (number of channels), depth (number of layers), [dropout layers](https://arxiv.org/abs/1207.0580), [L2 regularization](https://dl.acm.org/doi/10.5555/1795114.1795128), and [batch normalization](https://arxiv.org/abs/1502.03167). In general, we found that using methods like dropout and batch normalization, we can help prevent overfitting obtain higher accuracies as a result. 

Next, we studied some more advanced CNN architectures. We first tried the [Channel Boosted CNN](https://arxiv.org/abs/1804.08528). Then, we tried the [Residual Neural Network](https://arxiv.org/abs/1512.03385) (ResNet) with differing amounts of blocks. 

Then, we tried the [AlexNet](https://dl.acm.org/doi/10.1145/3065386) with moderate success. We improved upon this by following the [ZFNet](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53) architecture which changes, among a few other things, the filter size and stride in the first two layers. Their implementation saw an improvement over the AlexNet and we see an improvement in performance on this dataset as well. 

Additionally, we used some pre-trained models as a comparison for what can be achieved in the best case. We tried [Inception_V3](https://pytorch.org/hub/pytorch_vision_inception_v3/) using just PyTorch and ResNet-18 with [fastai](https://docs.fast.ai/). 

## Conclusion

Ultimately, we were able to find success with the ZFNet, achieving 99% training accuracy and 98% validation accuracy which is an improvement over the aforementioned gold medal winner on this dataset. 

![image](https://github.com/user-attachments/assets/3d670cc6-f292-4385-8796-625e036abbe5)

Above is a **confusion matrix** on the validation set (of 3,000 total images) which summarizes the nature of the misclassifications using this ZFNet model. Additionally, the pre-trained models were able to obtain a 100 percent training and validation accuracy, just like the gold medal winners. 

Along the way, we learned a lot about convolutional neural networks, training deep learning models, and how to search for parameters that are important. We also learned how to build a model from just basic PyTorch components as well as how to implement pre-trained models that can be transfer learned with the dataset at hand.

If this project were to be extended, we could do the following 
- Use the deconvolutional layers of the ZFNet to visualize the activations of the convolutional layers
- Train on a more complicated dataset that cannot achieve 100% performance on a pre-trained model.
- Make optimizations to the training that reduce the training time or number of epochs needed to train.

## Acknowledgements

We made extensive use of Kaggle's free 30 hours per week of GPU compute time for our analysis. Additionally, we thank the Erdos Institute for providing us the opportunity to learn about deep learning and work on this project.

- [PyTorch implementation of ZFNet](https://github.com/CellEight/Pytorch-ZFNet/tree/main)
- [PyTorch implementation of Local Contrast Norm](https://github.com/dibyadas/Visualize-Normalizations/blob/master/LocalContrastNorm.ipynb)
