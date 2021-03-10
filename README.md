# HRCNet: High-Resolution Context Extraction Network for Semantic Segmentation of Remote Sensing Images
This repository is a PyTorch implementation for our [Remote Sensing paper](https://www.mdpi.com/2072-4292/13/1/71).  
![HRCNet Framework](hrcnet.png)

## Usage  
### Version
pytorch >=1.1, we advise torch1.6.  
Training
```
sh auto_train.sh
```
Testing
```
python test.py
```
Restore image
```
python produce_result.py
```
Count metrics
```
python count_metrics.py
```
