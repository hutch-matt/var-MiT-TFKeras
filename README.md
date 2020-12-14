# Model-Zoo-v2

This is a collection of off-the-shelf and state-of-the-art deep learning TensorFlow-Keras models trained on the [Moments in Time](http://moments.csail.mit.edu/) dataset.  While none made huge breakthroughs, due to the computational costs of training these models on such a large dataset, we would like to make them available to others studying similar problems where transfer learning might be applicable.

## Contents

This directory contains:
 - a folder of trained models
 - main.py - the training/validation python script
 - utils.py - support methods for training/validation
 - example_build.ipynb - instructions and examples for working with and modifying these models
 - example_inference.ipynb - a single video example of parsing and inference with the trained models
 - example_train.sh - an example shell script for training with main.py

## Model Overview

Three types of "off-the'shelf" models are included in this "Model Zoo": 2D CNNs ([C2D](https://en.wikipedia.org/wiki/Convolutional_neural_network#Image_recognition)), 3D CNNs ([C3D](https://arxiv.org/pdf/1412.0767.pdf), [one-stream I3D](https://arxiv.org/abs/1705.07750)), and an [LRCN](https://arxiv.org/pdf/1411.4389.pdf) (CNN+LSTM).

C2D models were trained by uniformly randomly sampling frames from the input video.  C3D, I3D, and LRCN models were trained by using 16 or 32 dense frame snippets uniformly randomly sampled from the input video.

### Naming Convention

Each of the models is named in the following way: (backbone_name)-(input_shape)-(output_classes)-(training_history).h5

### Descriptions

The original backbones from which these "off-the-shelf" models were created and trained are linked to in the table below.  The following is a brief description of the models:

| Model | Name | Input Shape | # Classes | Training History |
| :----- | :----- | :----- | :----- | :-----|
| C3D-16x224x224x3-339-m.h5 | [C3D](https://github.com/axon-research/c3d-keras) ([source license](https://github.com/axon-research/c3d-keras/blob/master/LICENSE.md)) | (16,224,224,3) | 339 | Moments in Time |
| D169-224x224x3-339-im.h5 | [DenseNet169](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet169) ([source license](https://www.apache.org/licenses/LICENSE-2.0)) | (224,224,3)  | 339  | ImageNet (pretrained weights set) <br/> Moments in Time  |
| D201-224x224x3-339-im.h5 | [DenseNet201](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet201) ([source license](https://www.apache.org/licenses/LICENSE-2.0)) | (224,224,3)  | 339  | ImageNet (pretrained weights set) <br/> Moments in Time|
| I3DIv1-16x224x224x3-339-ikm.h5 | [Inflated Inception-v1 3D ConvNet](https://github.com/deepmind/kinetics-i3d) ([source license](https://github.com/deepmind/kinetics-i3d/blob/master/LICENSE)) | (16,224,224,3)   | 339   | ImageNet and Kinetics (pretrained weights sets) <br/> Moments in Time  |
| I3DIv1-32x224x224x3-339-ikm.h5 | [Inflated Inception-v1 3D ConvNet](https://github.com/deepmind/kinetics-i3d) ([source license](https://github.com/deepmind/kinetics-i3d/blob/master/LICENSE)) | (32,224,224,3)   | 339   | ImageNet and Kinetics (pretrained weights sets) <br/> Moments in Time  |
| IRv2-224x224x3-339-ikm.h5 | [Inception-ResNet-v2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionResNetV2) ([source license](https://www.apache.org/licenses/LICENSE-2.0)) | (224,224,3)  | 339  | ImageNet (pretrained weights set) <br/> Moments in Time   |
| IRv2avg-64x224x224x3-339-ikm.h5 | [Inception-ResNet-v2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionResNetV2) ([source license](https://www.apache.org/licenses/LICENSE-2.0)) | (64,224,224,3)  | 339  | ImageNet (pretrained weights set) <br/> Moments in Time   |
| Iv3-224x224x3-339-im.h5 | [Inception-v3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3) ([source license](https://www.apache.org/licenses/LICENSE-2.0)) |(224,224,3)   | 339   | ImageNet (pretrained weights set) <br/> Moments in Time  |
| LRCN-16x224x224x3-339-m6h5 | [Long-term Recurrent Convolutional Network](https://github.com/harvitronix/five-video-classification-methods/blob/master/models.py) ([source license](https://github.com/harvitronix/five-video-classification-methods/blob/master/LICENSE)) | (16,224,224,3)  | 339  | Moments in Time  |
| M-224x224x3-339-im.h5 | [MobileNet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNet) ([source license](https://www.apache.org/licenses/LICENSE-2.0)) | (224,224,3)   | 339   | ImageNet (pretrained weights set) <br/> Moments in Time  |
| Mv2-224x224x3-339-im.h5 | [MobileNet-v2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2) ([source license](https://www.apache.org/licenses/LICENSE-2.0)) | (224,224,3)  | 339  | ImageNet (pretrained weights set) <br/> Moments in Time  |
| R50-224x224x3-339-im.h5  | [ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50) ([source license](https://www.apache.org/licenses/LICENSE-2.0)) | (224,224,3)  | 339  |ImageNet (pretrained weights set) <br/> Moments in Time   |
| VGG19-224x224x3-339-im.h5 | [VGG19](https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG19) ([source license](https://www.apache.org/licenses/LICENSE-2.0)) | (224,224,3)  | 339  | ImageNet (pretrained weights set) <br/> Moments in Time  |
| X-224x224x3-339-im.h5 | [Xception](https://www.tensorflow.org/api_docs/python/tf/keras/applications/Xception) ([source license](https://www.apache.org/licenses/LICENSE-2.0)) |(224,224,3)   |339  | ImageNet (pretrained weights set) <br/> Moments in Time|

## Dependencies

 - Python 3.6.9
 - Horovod 0.18.2
 - OpenMPI 4.0
 - NumPy 1.16.5
 - H5Py 2.9.0
 - SciPy 1.3.2
 - TensorFlow 1.14.0

## Questions

Any questions can be directed to Matthew Hutchinson at <hutchinson@alum.mit.edu>.