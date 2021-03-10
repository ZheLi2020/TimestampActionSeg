# Temporal Action Segmentation from Timestamp Supervision

This repository provides a PyTorch implementation of the paper Temporal Action Segmentation from Timestamp Supervision.

Tested with:

- PyTorch 1.1.1
- Python 3.6.10
  
### Training:
* Download the data folder, which contains the features and the ground truth labels. (~30GB) (try to download it from [here](https://zenodo.org/record/3625992#.Xiv9jGhKhPY)))
* Extract it so that you have the 'data' folder in the same directory as 'main.py'.
* Put the timestamp annotations into the ground truth folder. For example, './data/breakfast/groundTruth/'.
* To train the model run 'python main.py --dataset=DS --split=SP' where 'DS' is 'breakfast', '50salads' or 'gtea', and 'SP' is the split number (1-5) for 50salads and (1-4) for the other datasets.

### Prediction and Evaluation:

Run `python main.py --action=predict --dataset=DS --split=SP`. 

### Model

The model used in this paper is a refined MS-TCN model. Please refer to the paper [MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://arxiv.org/pdf/1903.01945.pdf).


### Citation:

If you use the code, please cite

    Zhe Li, Yazan Abu Farha and Juergen Gall.
    Temporal Action Segmentation from Timestamp Supervision.
    In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021
    
    Y. Abu Farha and J. Gall.
    MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation.
    In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019
