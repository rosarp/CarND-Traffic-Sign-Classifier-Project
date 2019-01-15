# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/barchart.png "Barchart Visualization"
[image2]: ./examples/training_data_set.png "Training data set example"
[image3]: ./examples/colorspace1.png "Colorspace & qualization"
[image4]: ./examples/colorspace1.png "Colorspace & CLAHE"
[image5]: ./examples/random_gen_img1.png "Random Generated image"
[image6]: ./examples/random_gen_img2.png "Random Generated image"
[image7]: ./examples/barchart2.png "Barchart Visualization"

[image8]: ./online-samples/2.png "Online samples"
[image9]: ./online-samples/4.png "Online samples"
[image10]: ./online-samples/23.png "Online samples"
[image11]: ./online-samples/24.png "Online samples"
[image12]: ./online-samples/31.png "Online samples"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/rosarp/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribution accross classes for training, validation & test sets. 

![alt text][image1]

Also displayed sample from training data set.
It shows many images in training data set has low light exposure.  

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I discovered there are two techniquies to equalize brightness, which is primary concern in training set.

1. equalizeHist
2. CLAHE

So, As a first step, I decided to test which combination of colorspace & applying equalization technique will benifit more.

Here is an example of a traffic sign image before and after applying techniques.

![alt text][image3] 
![alt text][image4]

I nocied YUV CLAHE image is better candidate than others for further processing.

I applied the technique in `apply_clahe` method.

As a last step, I converted the image to grayscale.

I used three different combinations to test NN against.

    -- Gray 1 channel     |  97.40% accuracy on training set
    -- RGB  3 channel     |  95.7%  accuracy on training set
    -- RGB+Gray 4 channel |  95.4%  accuracy on training set


I decided to generate additional data because NN requires large data and current data set is only 34799.

To add more data to the the data set, I used the following techniques.

    -- PERTURBED_POSITION = (-2, 2)
    -- SCALE_RATIO = (0.9,1.1)
    -- ROTATION = (-15, 15)

Randomly generating images, so that NN can learn better from the variations in the set.

I saved all the sets in pickle files locally.

Here is an example of an original image and an augmented image:

![alt text][image5]
![alt text][image6]

The difference between the original data set and the augmented data set is the following ... 

I generated 7.5 times the count of each type of class of image. Thus total set 260992 plus original training set of 34799. 
Which in trun is 295791 samples in total for NN to learn from.

![alt text][image7]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride,  outputs 10x10x16  				|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten    			| outputs 400  									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer, batch size 128, epochs 100 & learning rate of 0.0001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I used gray images for training the model.
My final model results were:
* training set accuracy of **97.4%**
* validation set accuracy of **97.4%**
* test set accuracy of **93.4%**

I focused on pre processing the image to improve performance.
Tried with 3 different approaches. 
1. Gray images with 1 channel
2. RGB images with 3 channel
3. RGB + gray images with 4 channel

The best performance was given by gray images with 97.4% accuracy.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
Lenet5 architecture was chosen. The parameters used were taken from `Pierre Sermanet and Yann LeCun` white paper.
http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image8] ![alt text][image9] ![alt text][image10] 
![alt text][image11] ![alt text][image12]

Test accuracy given was 80% on this set. The last image might be difficult because of rough similarity in structure of bicycles & animal jumping, but it classified closely to the crossing sign. 


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image							|	 Prediction  								| 
|:-----------------------------:|:---------------------------------------------:| 
| Speed limit (50km/h) 			| Speed limit (50km/h)   						| 
| Speed limit (70km/h) 			| Speed limit (70km/h) 							|
| Slippery road  				| Slippery road 								|
| Road narrows on the right 	| Road narrows on the right 	 				|
| Wild animals crossing 		| **Bicycles crossing**      					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

The accuracy on the test set predicted with 93.4% accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the section named `Output Top 5 Softmax Probabilities For Each Image Found on the Web` of the Ipython notebook.

For the first image, the model is relatively sure that this is a `Speed limit (50km/h)` sign (probability of 0.99), and the image does contain a `Speed limit (50km/h)` sign. The top five soft max probabilities were
![alt text][image8]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (50km/h)  						| 
| .00     				| Speed limit (30km/h) 							|
| .00					| Speed limit (80km/h) 							|
| .00	      			| Speed limit (60km/h) 			 				|
| .00				    | Bicycles crossing    							|

For the second image, the model is relatively sure that this is a `Speed limit (70km/h)` sign (probability of 0.99), and the image does contain a `Speed limit (70km/h)` sign. The top five soft max probabilities were
![alt text][image9]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (70km/h)  						| 
| .00     				| No vehicles 									|
| .00					| Speed limit (30km/h) 							|
| .00	      			| Speed limit (50km/h) 			 				|
| .00				    | Speed limit (20km/h) 			 				|

For the third image, the model is relatively sure that this is a `Slippery road` sign (probability of 0.97), and the image does contain a `Slippery road` sign. The top five soft max probabilities were
![alt text][image10]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97         			| Slippery road   								| 
| .024     				| Priority road 								|
| .003					| Bicycles crossing 							|
| .00	      			| Wild animals crossing 		 				|
| .00				    | Double curve 									|

For the fourth image, the model is relatively sure that this is a `Road narrows on the right` sign (probability of 0.98), and the image does contain a `Road narrows on the right` sign. The top five soft max probabilities were
![alt text][image11]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| Road narrows on the right  					| 
| .008     				| General caution 								|
| .004					| Traffic signals 								|
| .00	      			| Road work 			 						|
| .00				    | Speed limit (30km/h)    						|

For the fifth image, the model is relatively sure that this is a `Bicycles crossing` sign (probability of 0.98), **but** the image contains a `Wild animals crossing` sign. The top five soft max probabilities were
![alt text][image8]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| Bicycles crossing  							| 
| .01     				| Wild animals crossing 						|
| .00					| Slippery road 								|
| .00	      			| Road work 			 						|
| .00				    | Beware of ice/snow    						|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


