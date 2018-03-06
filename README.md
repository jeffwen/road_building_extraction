# Road Extraction Using PyTorch

[//]: # (Image References)

[sat_img_banner]: etc/sat_img_banner.png "Satellite Image Banner"
[final_output]: etc/final_output.png "Final Output"
[unet_arch]: etc/unet_arch.png "UNet Architecture"
[train_img_overlay]: etc/train_img_overlay.png "Final Output"
[tensorboard]: etc/tensorboard.gif "Tensorboard"

![alt text][sat_img_banner]

This repository contains the code used for [this project](http://jeffwen.com/2018/02/23/road_extraction). Feel free to click on the link to read more about the details of the project!

The main dataset used to train the network was the Road and Building Detection [Dataset](https://www.cs.toronto.edu/~vmnih/data/) created by Volodymyr Mnih as part of his PhD thesis at the University of Toronto. Below is an example of a training image overlayed with the road network mask.

![alt text][train_img_overlay]

The main architecture is the [U-Net architecture](https://arxiv.org/pdf/1505.04597) that was used initially for biomedical image segmentation, but has since been successful in other tasks such as satellite imagery segmentation.

![alt text][unet_arch]

During the training process, I mainly used Tensorboard to keep track of the different runs.

![alt text][tensorboard]

Below is an example of the output from the network (the left image is the input RGB satellite image, the middle is the ground truth label, and the right side shows the model prediction).

![alt text][final_output]