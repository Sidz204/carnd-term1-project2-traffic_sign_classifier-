# **Traffic Sign Recognition** 

## Overview

### In this project, deep neural networks and convolutional neural networks are used to classify traffic signs. Specifically,we will train a model to classify traffic signs from the German Traffic Sign Dataset and later use this model to predict the sign in the images from web.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
You're reading it! and here is a link to my [project code](https://github.com/Sidz204/carnd-term1-project2-traffic_sign_classifier-/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the python and numpy libraries to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is disributed across each class.

![visualization](/visualizations/index1.png)


### Design and Test a Model Architecture

#### 1. Data Pre-processing:

1. Through exploratory visualization, I found out that some of classes has low image data which will not help the model to predict these signs as there was little training done on these datasets. Hence I found all the labels which has datasets count less than 500 and applied following augmentation techniques on such datasets:
    - Brightness
    
    ![augmentation](/visualizations/aug1.png)
    
    - Rotation
    
    ![augmentation](/visualizations/aug2.png)
    
    - Translation
    
    ![augmentation](/visualizations/aug3.png)
    
    - Shear
    
    ![augmentation](/visualizations/aug4.png)
    
The difference between the original data set and the augmented data can be shown with the following graphs:

![visualization](/visualizations/index1.png)
![visualization](/visualizations/index2.png)


2. Grayscaling and normalization:
Later, I grayscaled the image data using weights(grayconver= np.array([[0.2989],[0.5870],[0.1140]])) as there is no need of colour to identify signs. Later, I normalize it using x=(x-128)/128 so all the data fall within equal mean and unit covariance.
Here is an example of a traffic sign image before and after grayscaling & normalizing.

![visualization](/visualizations/grayscale.png)
 


#### 2. Description of Final model architecture


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscaled image   					| 
| 1.Convolution 3x3    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| 												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6   				|
| 2.Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					| 												|
| Max pooling	      	| 2x2 stride, outputs 5x5x16     				|
| Flatten    	      	| Outputs: 400                   				|
| 3.Fully connected		| outputs: 120     					     		|
| RELU                  |                                               |
| 4.Fully connected  	| Outputs: 84  									|
| RELU					|												|
| 5.Fully connected		| Outputs: 43									|
 


#### 3. Model training : 

I used an Adam optimizer, batch size of 128 for training. I tried epochs for 50 ,60 ,80, 100 and found out that 80 gives best results. Learning rate was set to 0.001. Also, tried dropouts with keep probability of 0.5,0.7,0.9 but later found that model performs better without dropouts.


#### 4. Architecture selection and Iterative approach

Architecture chosen:
I choose LeNet architecture as it has been widely used for recognizing handwritten digits(MNIST data) and gives good accuracy with decent amount of computation power. I think it should give good accuracy for traffic sign recognition with some modifications.


Iterative approach:
- First I tried LeNet architecture on rgb inputs without normalization, epochs 10, batchsize 128, maxpooling and RELU activation function. I got validation accuracy of 89%.

- Later I just changed the epochs to 60 and grayscaled (single channel) and normalized input data. Accuracy increased to 95%.But while testing on new images , images that fall under the category/label which has less training data gave wrong results.

- This time I augmented the data with labels which has low data count. I applied rotation, translation, brightness and shear techniques on each image data. Also added dropouts with keep probability of 0.5. I got accuracy of 90%.

- Later, I tried increasing epochs to 80 and 100 and removing droputs. Then I noticed it gives good results in 80 itself. I got a accuracy of 94.8% 


My final model results were:
* validation set accuracy of 94.8%
* test set accuracy of 92.5%
* accuracy on new images : 80% (8 out of 10 images correctly predicted)

### Test a Model on New Images

#### 1. German traffic signs images found on the web

I found 10 German traffic sign images on the web. Here are five of them:

![image1](/new_images/30km.jpg) 
![image2](/new_images/slippery.jpg) 
![image3](/new_images/img2.jpg) 
![image4](/new_images/stop.jpg) 
![image5](/new_images/rightoff.jpg)

I resized each of these images into 32x32x3 image.The second image might be difficult to classify because it was taken from a wrong angle so only a smaller part of sign is visible and sometimes human will not be able to predict it correctly at first look. Rest of the images gave accurate results.

#### 2. Model's predictions on the new traffic signs

Here are the results of the prediction of 5 images:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30km Speed limit     	| 30km Speed limit   							| 
| Slippery road     	| Road narrows on right							|
| No entry				| No entry  									|
| Stop  	      		| Stop      					 				|
| Right-off way			| Right-off way      							|


The model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 80%.

#### 3. Top 5 softmax probabilities for each image along with the sign type of each probability.

The code for the Top 5 softmax probabilities for each image is there in the Ipython notebook.

For the first image, the model is pretty sure that this is a stop sign (probability of 0.99), and the image does contain a stop sign. Similarly,other top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| 30km speed limit   				    		| 
| .50     				| Road narrows on right 						|
| .99					| No entry	    								|
| 1.0	      			| Stop					        				|
| 1.0				    | Right-off way      							|


For the second image ... 


