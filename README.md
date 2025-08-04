# LuKAN
**A Kolmogorov-Arnold Network Framework for 3D Human Motion Prediction**

This repository contains the code for our project on 3d human motion prediction.

### Architecture

![LuKAN Architecture](Figures/Figure1.jpg)

### Requirements
------
- matplotlib==3.9.1
- numpy==1.26.4
- scikit_learn==1.5.1
- setuptools==69.5.1
- sympy==1.12.1
- torch==1.13.0
- tqdm==4.66.4
- pandas==2.2.2
- seaborn
- pyyaml==6.0.1
- PyWavelets==1.6.0
- einops==0.8.0
- scipy==1.13.0
- six==1.16.0
- Easydict==1.13


### Datasets
------
All of our datasets were taken from their official websites.

- [H3.6M](http://vision.imar.ro/human3.6m/description.php)

- [AMASS](https://amass.is.tue.mpg.de/)

- [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)

### Commands
------
#### H3.6M
##### Training
```bash
cd h36m
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py
```
##### Evaluation
```bash
python test.py --model-pth your/model/path
```

#### AMASS AND 3DPW
##### Training
```bash
cd amass_and_3dpw
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py
```
##### Evaluation
```bash
#Test on AMASS
python test.py --model-pth your/model/path 
#Test on 3DPW
python test_3dpw.py --model-pth your/model/path 
```

## 📖 Citation

If you use this work, please cite:

```bibtex
@inproceedings{zadid2025lukan,
  title={LuKAN: A Kolmogorov-Arnold Network Framework for 3D Human Motion Prediction},
  author={Hasan, Md Zahidul and Ben Hamza, Abdessamad and Bouguila, Nizar},
  booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
  year={2025}
}

