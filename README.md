# Traffic

## Description and Data set
This project implements a convolutional neural network to identify which traffic sign appears in a photograph.

The data set used is provied by:
J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011,
which can be downloaded from **[here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)**

## Walkthrough

At first I tried the convolutional network shown in CS50AI 2020 lecture 5 for the CNN. This CNN had:
A. One convolutional Layer that learns 32 filters using 2*2 kernels
B. One max-pooling layer using 2*2 pool size
C. 1 hidden layer with 128 nodes and a "relu" activation function
D. dropout of 0.5 to avoid overfitting 
E. An ouput layer with output units for all number of categoreies

This implementation of the CNN network did not prouduce satsifactory results on big datasets, accuaracy was around 60 %.

Afterwards, I tried to increase the number of filter the CNN learns to 100 instead of 32, to my surprise this made the results even worse. 


Finally I decided to repeat my layers of convolution and pooling in addition to increasing the number of nodes in the hidden layer and reached a pretty satsifactory results (between 94% and 96%)

This is my **final structure** of the CNN:

A. A convolutional layer  that learns 32 filters using 3*3 kernels
B. A max-pooling layer using 2*2 pool size
C. A second convolutional layer similar to the first
D. A second pooling layer similar to the first
E. One hidden layer with 512 node and a dropout of 0.5 to avoid overfitting
E. An ouput layer with output units for all number of categoreies

