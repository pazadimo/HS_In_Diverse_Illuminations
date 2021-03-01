# HS_In_Diverse_Illuminations
This repository contains Dataset and code for training and testing the deep neural network proposed in:
**"Enabling Hyperspectral Imaging in Diverse IlluminationConditions for Indoor Applications"**

# Hyperspectral Reconstruction from RGB Images for Vein Visualization
Hyperspectral imaging provides rich information across many wavelengths of the captured scene, which is useful for many potential applications such as food quality inspection, medical diagnosis, material identification, artwork authentication, and crime scene analysis. However, hyperspectral imaging has not been widely deployed for such indoor applications. In this work, we address one of the main challenges stifling this wide adoption, which is the strict illumination requirements for hyperspectral cameras. Hyperspectral cameras require a light source that radiates power across a wide range of the electromagnetic spectrum. Such light sources are expensive to setup and operate, and in some cases, they are not possible to use because they could damage important objects in the scene. We propose a data-driven method that enables indoor hyperspectral imaging using cost-effective and widely available lighting sources such as LED and fluorescent. These common sources, however, introduce significant noise in the hyperspectral bands in the invisible range, which are the most important for the applications. Our proposed method restores the damaged bands using a carefully-designed supervised deep-learning model. We conduct an extensive experimental study to analyze the performance of the proposed method using real hyperspectral datasets that we have collected. Our results show that the proposed method outperforms the state-of-the-art across all considered objective and subjective metrics, and it produces hyperspectral bands that are close to the ground truth bands captured under ideal illumination conditions.

## Dataset Structure
- Download link -  https://nsl.cs.sfu.ca/projects/hyperspectral/hyperspectral_data/dataset.zip
- The dataset consists of paired 207 RGB images with their corresponding hypercubes in total.
- The hyperspectral images contain 34 bands in spectral range 820-920nm in matlab (`.mat`) format extracted from raw data.
- The total dataset is having information (images) from 13 participants. 10 participants' data is used for training and remaining 3 participants' data is used for testing/validation.
- Folder contents: The downloaded folder contains a sub-directory named `veins_t34bands`, further having dataset divided into `train_data`, `valid_data` and `test_data` folders. Each dataset folder is further divided into `mat` and `rgb` folders having hyperspectral and RGB images respectively.

## Source Code
### Prerequisites
- Linux or macOS
- Python 3
- Pytorch
- NVIDIA GPU + CUDA CuDNN
- MATLAB

### Installation
- Clone this repo:
```bash
git clone https://github.com/pazadimo/HS_In_Diverse_Illuminations.git
cd vein-visualization
```
- Install [PyTorch](http://pytorch.org) and other dependencies.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.



### Train  


#### Dataset preparation
- The model requires 50x50x25 patches sampled in `.h5` from augmented dataset for the training process.
- You can either use our `.h5` files containg sampled patches or you can sample your own patches and save them `.h5` file.
- Original Augmented dataset in `.mat` format for the Fruit Processing and Material Identification is available at [Fruit](https://drive.google.com/drive/folders/1BI6J3aJiuqpXMFlNwYt3O0JLP3PHW4zD?usp=sharing) and [Material](https://drive.google.com/drive/folders/1LBvEqoJuQ3o9ryulqWbktEmI3K-g-K_1?usp=sharing)
- You can should put these two folders in `./train/Data` folder and use `save_h5_data_fruit.m` and `save_h5_data_material.m` for making `.h5` files. However, we recommend you to use our `.h5` files instead of rebuilding them.
- Training and Validation `.h5` files for the both Fruit Processing and Material Identification categories are available at: [Fruit Processing Training](https://drive.google.com/file/d/1qQGmerp7RU6igRSg7gUWX62EvTj1YYsS/view?usp=sharing), [Fruit Processing Validation](https://drive.google.com/file/d/1EvY3f-Rbm2FYMmw7SWA30pbO4WyTWXqz/view?usp=sharing), [Material Identification Training](https://drive.google.com/file/d/1fhotXS85J7Bt1oH8AHxa4zNt9fon1wJt/view?usp=sharing), and [FMaterial Identification Validation](https://drive.google.com/file/d/1_hZJZIYA2yI0v2WRkpIFpur6ae8ldCup/view?usp=sharing).
- Move downloaded dataset folder to root (vein-visualization/dataset)
- Increase the training data using augmentation techniques (rotation and flipping). The matlab file `./train/augment_data.m` is used to perform augmentaion.
- The dataset is stored in HDF5 (`.h5`) file for training process. The matlab file `./train/generate_paired_rgb_nbands.m` is used to generate `train.h5` and `valid.h5` dataset files.



The training and testing codes are present in `./train/` and `./test/` folders respectively. The model architecture is present in `resblock.py` file.
- Train a model:
```bash
#!./train/train.py
python train.py
```
- The trained models will be stored in `./train/models/` folder with log files. 










### Test the model:
```bash
#!./test/evaluate_model.py
python evaluate_model.py
```
- The pre-trained models are present in `./test/models/`. The model can be evaluated on the testing dataset present in `./dataset/test_data/rgb/`. The test results will be saved to the folder: `./dataset/test_data/inference/`.




### Vein enhancement
- The reconstructed and ground truth hyperspectral images can be visualized in MATLAB using commands: `load(‘y.mat’);`,`imshow(rad(:,:,1),[]);`
- The reconstructed band can be enhanced using two enhancement techniques: Contrast Limited Adaptive Histogram Equalization (CLAHE) and Homomorphic Filtering.
- Enhancement can be produced using file `./vein_enhancement/enhance.m`.



