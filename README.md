# Temporal Action Segmentation from Timestamp Supervision

This repository provides a PyTorch implementation of the paper Temporal Action Segmentation from Timestamp Supervision.

Tested with:

- PyTorch 1.1.1
- Python 3.6.10
  
### Training:
* Download the data folder, which contains the features and the ground truth labels. (~30GB) (try to download it from [here](https://zenodo.org/record/3625992#.Xiv9jGhKhPY)))
* Extract it so that you have the data folder in the same directory as main.py.
* To train the model run python main.py --dataset=DS --split=SP where DS is breakfast, 50salads or gtea, and SP is the split number (1-5) for 50salads and (1-4) for the other datasets.
