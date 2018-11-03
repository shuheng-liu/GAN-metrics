"""This is an TensorFLow implementation of AlexNet by Alex Krizhevsky at all.

Paper:
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Explanation can be found in my blog post:
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

This script enables finetuning AlexNet on any given Dataset with any number of
classes. The structure of this script is strongly inspired by the fast.ai
Deep Learning class by Jeremy Howard and Rachel Thomas, especially their vgg16
finetuning script:
Link:
- https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16.py


The pretrained weights can be downloaded here and should be placed in the same
folder as this file:
- http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

@author: Frederik Kratzert (contact: f.kratzert(at)gmail.com)
"""

from .alexnet import AlexNet
from .siamese_alexnet import SiameseAlexNet

__all__ = [
    AlexNet,
    SiameseAlexNet,
]
