# HRCNet: High-Resolution Context Extraction Network for Semantic Segmentation of Remote Sensing Images
This repository is a PyTorch implementation for our [Remote Sensing paper](https://www.mdpi.com/2072-4292/13/1/71).  
<img src="hrcnet.png" width="800" height="400" alt="HRCNet Framework"/><br/>
## Usage  
Version 
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
## Acknowledgments
Thanks the ISPRS for providing the Potsdam and Vaihingen datasets.
## Citation
```
@article{xu2020hrcnet,
  title={HRCNet: high-resolution context extraction network for semantic segmentation of remote sensing images},
  author={Xu, Zhiyong and Zhang, Weicun and Zhang, Tianxiang and Li, Jiangyun},
  journal={Remote Sensing},
  volume={13},
  number={1},
  pages={71},
  year={2020},
  publisher={MDPI}
}
```
## Other Links
* [Efficient Transformer for Remote Sensing Image Segmentation](https://github.com/zyxu1996/Efficient-Transformer)
* [CCTNet: Coupled CNN and Transformer Network for Crop Segmentation of Remote Sensing Images](https://github.com/zyxu1996/CCTNet)
