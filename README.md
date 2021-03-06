# Temporal Action Segmentation from Timestamp Supervision

This repository provides a PyTorch implementation of the paper Temporal Action Segmentation from Timestamp Supervision.

Tested with:

- PyTorch 1.1.1
- Python 3.6.10
  
### Training:
* Download the [data](https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8) folder, which contains the features and the ground truth labels. (~30GB) (If you cannot download the data from the previous link, try to download it from here)
* Extract it so that you have the data folder in the same directory as main.py.
* To train the model run python main.py --action=train --dataset=DS --split=SP where DS is breakfast, 50salads or gtea, and SP is the split number (1-5) for 50salads and (1-4) for the other datasets.
