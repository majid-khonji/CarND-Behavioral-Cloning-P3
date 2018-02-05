# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model is based on Nvidia model, shown in the figure below:

![alt text](./examples/nvidia-arch.png "Nvidia Architecture" )







However, I slightly modified it to reduce overfitting (as discussed later below), with the following architecture (model.py lines 72-85) :

``` 
Layer (type)                     Output Shape          Param #     Connected to                     
=======================================================================================
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           cropping2d_input_1[0][0]         
_______________________________________________________________________________________
lambda_1 (Lambda)                (None, 90, 320, 3)    0           cropping2d_1[0][0]               
_______________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 43, 158, 24)   1824        lambda_1[0][0]                   
_______________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 77, 36)    21636       convolution2d_1[0][0]            
_______________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 37, 48)     43248       convolution2d_2[0][0]            
_______________________________________________________________________________________
dropout_1 (Dropout)              (None, 8, 37, 48)     0           convolution2d_3[0][0]            
_______________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 6, 35, 64)     27712       dropout_1[0][0]                  
_______________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 4, 33, 64)     36928       convolution2d_4[0][0]            
_______________________________________________________________________________________
dropout_2 (Dropout)              (None, 4, 33, 64)     0           convolution2d_5[0][0]            
_______________________________________________________________________________________
flatten_1 (Flatten)              (None, 8448)          0           dropout_2[0][0]                  
_______________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           844900      flatten_1[0][0]                  
_______________________________________________________________________________________
dropout_3 (Dropout)              (None, 100)           0           dense_1[0][0]                    
_______________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_3[0][0]                  
_______________________________________________________________________________________
dropout_4 (Dropout)              (None, 50)            0           dense_2[0][0]                    
_______________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_4[0][0]                  
_______________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
=======================================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
_______________________________________________________________________________________
```



To summarize, the model is structured as follows:

* Three sets of convolution layers, each set with 5x5 kernel and strides of 2x2 and depth of 3,24, 36, respectively. Each convolution layer is followed by RELU activation (to introduce non-linearity).
* Dropout layer with a probability of .2 (to reduce overfitting)
* Two sets of convolution layers, each with 3x3 kernel and strides of 1x1 and depth of 64
* Dropout  layer with a probability of  0.15
* Dense layer with 100 outputs
* Dropout layer with a probability of  0.1
* Dense layer with 50 outputs
* Dropout layer with a probability of  0.05
* Dense layer with 10 outputs
* Dense layer with 1 outputs

The model includes preprocessing steps: cropping images (model.py line 68), and normalization (model.py line 70).

#### 2. Attempts to reduce overfitting in the model

I observed training loss is a bit smaller than validation loss through epochs, therefore I introduced Dropouts (model.py, lines 75,78,81, 83). 

I attempted different number of Dropout layers and probabilities. Each attempt showed either overfitting or underfitting (with no effect). For instance, I tried an a Dropout after each layer with a a probability of $0.3$. However, this model showed signs of overfitting.

Eventually, Dropouts were placed after each of the two sets of convolution layers and Dense layers, except he last two Dense layers, with descending probabilities. The rationale behind is that we want the model to "forget more" low level features than higher level features. We obtained similar loss between training and validation sets, while maintaining low values of loss.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 106).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road in Track 1. I observed that training using the keyboard doesn't provide smooth driving and often ends up hitting road boundaries and eventually the vehicle exists the road. Therefore, I used the touch pad to give smooth commands, but keyboard to maintain center location. I also collect data in the opposite direction to make sure the model generalizes well and avoids bias towards left steering commands.

I considered the following scenarios in the final data set:

* 1 lap using the keyboard, making sure the vehicle is on the center
* 2 laps using the touch pad with smooth driving
* 1 lap in the opposite direction using the touch pad

Then I observed where the vehicle doesn't perform well (e.g., falling in water before the bridge). I recorded training data that deals with these scenarios multiple times.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a well tested model used in a similar application. A good starting point is Nvidia model.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80% and 20% respectively). I found that the original Nvidia model had a low mean squared error on the training set but a higher mean squared error on the validation set. This implied that the model was overfitting. Therefore I introduced Dropouts (model.py, lines 75,78,81, 83). The Dropouts were placed after each of the two sets of convolution layers and Dense layers, except he last two Dense layers, with descending probabilities. The rationale behind is that we want the model to "forget more" low level features than higher level features. We obtained similar loss between training and validation sets, while maintaining low values of loss.

After using Dropouts, training and validation losses were quite close (as shown the last figure), while maintaining low values of loss. I also converted the image color from BGR to RGB (model.py, line 44). This showed a significant improvement on the result (I don't known exactly why, but it worked!). 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (fell in water before the bridge, and on the long road). To improve the driving behavior in these cases, I recording training data from edges of the track so that the algorithm learns what to do in such scenarios.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is illustrated above.

The code has the following main functions:

* `process_csv_data`: reads data from the csv file, updates image directory prefix, updates steering angles for the right and left camera images; and returns image addresses and steering angles. 
* `generator`: reads image addresses and steering addresses, and return RGB image and labels as batches (loaded to memory on demand)
* `arch_nvidia`: returns the updated Nvidia model
* `train`: contains the full training pipeline, saves the model file and loss values

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving using the touchpad and one lap with the keyboard. Here is an example image of center lane driving:

![alt text](./examples/2018-02-05-00:14:42_3200x1800_scrot.png "Training" )

I then recorded the vehicle recovering from places where the vehicle exists the road, as illustrated above.

After the collection process, I had 9958 number of data points (and  29874 images). I then preprocessed this data by renaming image directory using that of the CSV file in order to maintain a correct directory address reading on the GPU instance in AWS (lines 8-29). The steering angle is also adjusted for the right and left camera images with a correction of $0.2$. The `generator` function (line 33) reads image names and angles and divides them into batches, loading 1 batch of images at a time into RAM, converts them to RGB, and returns the complete data set  (`X_train`) and labels (`y_train`) for the batch.


I finally randomly shuffled the data set and put 20% of the data into a validation set. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. A reasonable number of epochs was 7 as evidenced by the plot below, as it shows a constant decrease in both losses at a similar rate. I noticed the model is a bit under fitting, because the training loss is lower than validation, but the loss is quite low for both, therefore I neglected this issue.  I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text](./examples/loss.jpg "Loss" )