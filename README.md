# Temporal Action Segmentation from Timestamp Supervision

This repository provides a PyTorch implementation of the paper [Temporal Action Segmentation from Timestamp Supervision.](https://arxiv.org/abs/2103.06669)

Tested with:

- PyTorch 1.1.0
- Python 3.6.10
  
### Training:
* Download the data folder, which contains the features and the ground truth labels. (~30GB) (try to download it from [here](https://zenodo.org/record/3625992#.Xiv9jGhKhPY)))
* Extract it so that you have the `data` folder in the same directory as `main.py`.
* The three `.npy` files in 'data/' in this repository are the timestamp annotations. Put each one in corresponding ground truth folder. For example, `./data/breakfast/groundTruth/` for Breakfast dataset.
* To train the model run `python main.py --action=train --dataset=DS --split=SP` where `DS` is `breakfast`, `50salads` or `gtea`, and `SP` is the split number (1-5) for 50salads and (1-4) for the other datasets.
* The output of evaluation is saved in `result/` folder as an excel file.
* The `models/` folder saves the trained model and the `results/` folder saves the predicted action labels of each video in test dataset.

### Prediction and Evaluation:

Normally we get the prediction and evaluation after training and do not have to run this independently.
In case you want to test the saved model again by prediction and evaluation, please change the `time_data` in `main.py` and run 
  
  `python main.py --action=predict --dataset=DS --split=SP`. 

### Model

The model used in this paper is a refined MS-TCN model. Please refer to the paper [MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://github.com/yabufarha/ms-tcn).


### Citation:

If you use the code, please cite

    Zhe Li, Yazan Abu Farha and Juergen Gall.
    Temporal Action Segmentation from Timestamp Supervision.
    In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021
    
