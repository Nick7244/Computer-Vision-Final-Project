# Computer Vision Final Project
## A Deep Learning-based Approach to Autonomous Vehicle Path Planning and Object Detection
Akaash Sanyal, Akash Chaurasia, Nicholas Maritato, Parth Vora

## Current Workflow
### Set up architectures
#### VGG16
- Update VGG_model.py
- Remove final layer and replace with correct classification
#### ResNet
- Update ResNet_model.py
- Remove final layer and replace with correct classification
#### InceptionV3
- Update Inception_model.py
- Remove final layer and replace with correct classification
### Custom
- Decide on custom architecture
- Update Custom_model.py

### Train models
- Using dataset, train relevant parts of model

### Test performance and speed
- Using dataset, test accuracy and speed of prediction

### Find optimal preprocessing
- Use feature data from 3 established architectures as input to custom architecture
- Summarize performance and speed

### Find optimal feature layer
- Use feature data from between blocks of established architectures as input to custom architecture
- Summarize performance and speed
