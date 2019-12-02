# Computer Vision Final Project
## A Deep Learning-based Approach to Autonomous Vehicle Path Planning and Object Detection
Akaash Sanyal, Akash Chaurasia, Nicholas Maritato, Parth Vora

## Current Workflow
### Set up architectures with transfer learning
- VGG16
  - Update VGG_model.py
  - Remove final layer and replace with correct classification
- ResNet
  - Update ResNet_model.py
  - Remove final layer and replace with correct classification
- InceptionV3
  - Update Inception_model.py
  - Remove final layer and replace with correct classification

### Train models
- Using dataset, train new top layer

### Test performance and speed
- Using dataset, test accuracy and speed of prediction

### Find optimal feature layer
- For preloaded architecture with the best performance, fine-tune model by unfreezing blocks
