# **Traffic Sign Recognition** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---
**Building a Traffic Sign Recognition Convolutional Neural Network**

![speed limit](images/traffic-sign-3008267_640.jpg)

The goals of the project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train, and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

- [Project Code](TrafficSignClassifier.ipynb)
## Data Set Summary
The data available for the training is the German traffic sign data set available from [The German Traffic Sign Benchmark challenge](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news). 

The images we will be using are color with a 32x32 size. The data is separated in training, validation, and test sets as follows:

- Training set samples: 34799
- Validation set samples: 4410
- Test set samples: 12630
- Images Size (32, 32, 3)
- 43 Sign Names 


### Data Exploration
To begin, we can see that the data is not evenly distributed on the following image:

![data distribution](images/distribution.jpg)

#### Random Samples Signs
To see some examples of the data we obtain random labels and their corresponding name from the `signnames.csv`file.

|Sample|Sign Id|Name|
|------|-------|----|
|204| 41 |End of no passing|
|6056| 3 |Speed limit (60km/h)|
|872| 31 |Wild animals crossing|
|8579| 4 |Speed limit (70km/h)|
|3452| 1 |Speed limit (30km/h)|
|4538| 22| Bumpy road|

#### Exploratory visualization
We also display a set of images with their correspondent labels since the data set is ordered.

![random images samples](images/random_samples.jpg)

## Design and Test a Model Architecture
The architecture selected is base on the Lenet-5 Network with 2 convolution layers followed by a max-pooling layer and 2 fully connected layers reduced to 43 outputs, to improve the accuracy we use a dropout operation before the fully connected layers.
![LeNet-5](images/LeNet5.png)
### Preprocessing
To improve the accuracy we process the images using the following strategies:

- Normalize color images dividing by 255
- Images converted to gray using skimage
- Add images with random noise
- Add rotation of 15 degrees right and left
- Shuffle the data

The most important step is to shuffle the data to give better results since we process the images with a batch of 128.

The training data is augmented from 34.799 to 139.000 images.

|Train data|Samples|
|---|----|
|Original| 34.799|
|Augmented data|104.397|
|Total |139.196|

### Model Architecture
We use the model from Pierre Sermanet and Yann LeCun: [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), Proceedings of International Joint Conference on Neural Networks (IJCNN'11), 2011. 

We modified the architecture as follows:

| number |Layer | Input Size| Ouput Size|Activation|
|--------|-------|-----------|-----------|----|
|0|Input| 32x32x1 | | |
|1|Convolution and max Pooling with stride of 1 and `valid` padding| 32x32x1| 16x16x16 |Relu|
|2|Convolution and max Pooling with stride of 1 and `valid` padding, Max Pooling| 16x16x16| 8x8x32 |Relu|
|3|Fully connected| 800 | 120 |Relu|
|4|Fully Connected | 120 | 43 |Relu|



### Model Training
The learning rate we use was **0.0005**, a dropout rate of **0.4**, a batch size of **128 elements**, and maximum **15 epochs**, however, we use and early stop when the accuracy we need is found **95%**, avoiding overfitting.

Finally using the test set we found accuracy is 0.935 which is what we were looking for.

## Test a Model on New Images
To see the certainty of the model's predictions we use a small set of internet images, presented here:

![custom images](images/custom.jpg)

Two images are not on the data set, used to see what could be the results, we give them some similar labels, but in any case, is difficult for the network to find a similar value.

We crop and resize the images to be used by the network to 32x32 and we obtain a test Accuracy of **67%**

The prediction in some cases has a high value but most of the time the network had low confidence in the found results. 

We present a summary here:

| Prediction | Predicted value | Expected Value |Other Prediction|
|------------|-----------------|----------------|-----------------|
|General caution (95%)|18|N/A| Road work (20%)| 
|Bumpy road (50%) |22|N/A|Bicycles crossing (49%)|
|Roundabout mandatory (76%)|40 |40 |End of no passing by vehicles over 3.5 metric tons (17%)|
|Right-of-way at the next intersection(99%)| 11 |11 |Beware of ice/snow (21%)|
|Children crossing (76%)| 28 | 28 |Bumpy road (78%)|
|Road work (100%) | 25 | 25 |Dangerous curve to the right (26%)|

## Conclusions
Using Tensorflow 1.x requires many implementation details that now days are hiding from the network developer using Keras. However, it is important to understand how the networks are designed. What effects every parameter has into the results. It is in any case time consuming to improve the precision and a known and simple architecture.

Having data augmentation was the most successful change in improving the accuracy of the network, it however requires a larger set to be able to identify other signals. For instance caution, risk, or information signs have specific shapes and colors, might be interesting to try to identify this first and the content of the sign.