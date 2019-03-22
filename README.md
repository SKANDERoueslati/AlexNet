# AlexNet
ready to use module of AlexNet CNN (TensorFlow implementation)

This project provides the AlexNet class that allows to directly integrate the AlexNet CNN by simplly instanciate the AlexNet class.

And then execute the following steps (an example will be added soon) :
1) Create the AlexNet object.
2) Construct the model by calling the construct() method.
3) Load the pretrained parameters using the loadTrainedParams() method, first the parameters file must be downloaded,
   it is can be found here http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ with the classes file also.
4) Assign the inputs
5) Run the model by running a tensorflow session on the cnn_output attribute of the model while providing the input feed.
