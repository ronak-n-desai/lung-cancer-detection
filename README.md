# Lung Cancer Detection with Histopathological Images

**Authors: Abuduaini Niyazi, Derek Kielty, Rishat Dilmurat, Sujoy Upadhyay, Ronak Desai**

*Submitted as a final project for the [Erdos Institute](https://www.erdosinstitute.org/) Deep Learning (Summer 2024) Boot Camp*

One of the most important applications of deep neural networks in the healthcare industry is the classification of various types of diseases from image data. Particularly, one life-saving application is to detect cancerous cells at an early stage. In this project, we will train convolutional neural networks with a [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) of 25,000 images to distinguish normal lung tissues from cancerous ones.

![image](https://github.com/user-attachments/assets/c89b7429-ab99-4678-9613-5fa1f08e6a53)

The above images are (from left to right) adenocarcinoma, squamous cell carcinoma, and normal lung tissue and are samples of images found in the data set. Not only do we need to differentiate cancer from no cancer, but we need to differentiate between the two types of lung cancer. 

[Here](https://www.geeksforgeeks.org/lung-cancer-detection-using-convolutional-neural-network-cnn/#) is an article that describes the [Kaggle Dataset of Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) and gives a basic approach to training the model with around 91 percent accuracy on a validation data set. Can we develop a model that can achieve higher accuracies? We explore various different architectures and several advanced CNN models and report our findings. Additionally, we compare these methods to our simple base-line model which had around 89 percent accuracy on the validation set to gain a better understanding of which modifications influence the accuracy the most.

