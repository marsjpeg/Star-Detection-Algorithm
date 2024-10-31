# Star-Detection-Algorithm
A novel approach to star detection using hough circle transform

## Description
Stars can be detected in images through the Hough circle transform, an image
transformation that allows for circular shapes to be detected in images. A common
issue with this, however, is false positives detected in noisy images that tend to
have multiple stars, along with other objects. The aim of this salgorithm is
to resolve the issue of detecting non-stars as stars by using a machine learning
model to make star detection more accurate. By training a Convolutional Neural
Network (CNN) model to detect between stars, non-star beings, and overall false
positives, the goal is to provide a new novel approach to star detection that can
detect stars in noisy, busy image environments. The result was a robust approach to
more accurate star detection through deep learning, a subfield of machine learning.

## Prerequisites
To run this project, make sure the following libraries are installed:

**OpenCV**:
  ```bash
  pip install opencv-python
```

**Tensorflow**:
  ```bash
  pip install tensorflow
```

Python 3.6 and higher is also required.
