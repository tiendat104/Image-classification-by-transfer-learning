# Image-classification-by-transfer-learning

# Abstract
In this project, i implement simple image classification task on custum data, using transfer learning for both keras and pytorch.

# Data preparation
Because the size of the whole dataset is big, so in this project, i only create a sample of few images for demo. You can get the whole dataset from this link: https://www.kaggle.com/puneet6060/intel-image-classification

After downloading the dataset from above links, please arange the images as the same as folder "data" in this project. There are 3 folders "train", "test", "val", each folder contains 6 folders that contains images from 6 classes.

# Environment requirement
- tensorflow==1.15.0
- keras==2.3.1
- torch==1.3.1

# Train and test 
# 1. Keras model 
Inside the file "keras/train.py" i write transfer learning for following model: MobileNet, Xception, Inception, VGG, Inception, ResNet50, DenseNet.  So, you can change the model by changing line 111 inside the function train(), for example, if i want to use DenseNet, i can simply change that line to 

model = transfer_learning_DenseNet(input_shape=image_shape, num_classes=num_classes)

Then, simply run this file and the model start training. 

After the training completed, weight would be saved to folder "keras/checkpoint" and training history will be saved to folder "keras/logs". 

To test trained model, then inside the file "keras/train.py", please change the weight path inside the test() function, and then call test() function inside the main function. 

Furthermore, i also implement transfer learning for EfficientNet. To use this model for transfer learning, you will need to install it by command "pip install efficientnet". 
Then to train it, simply run file "keras/EfficientNet_train.py" 

# 2. Pytorch model 
To train, simply run the file "pytorch/train.py".

Inside this file, i write transfer learning for ResNet and VGG models.

To switch model, simply change the line 86 in function train(). 

After training completed, a new folder "keras/checkpoint" would be created and weight would be saved into this folder. 

To test, then please pass the weight path to the test() function, and then call test() function inside the main function. 



