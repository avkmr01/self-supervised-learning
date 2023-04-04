# self-supervised-learning

##simClr
There are three components which are useful in simCLr fodler:
1. training.py --> It is the base for contrastive learning. It uses a large amount of dataset to extract the differential features from image.
2. finetuning.py --> It is used after extracting the features and narrow down to your required label.
3. direct_training.py --> without extracting any features directly do the supervised learning.
