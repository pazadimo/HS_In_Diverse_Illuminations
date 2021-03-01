# HS_In_Diverse_Illuminations
This repository contains the dataset and code used for training and testing the deep neural network proposed in:
**"Enabling Hyperspectral Imaging in Diverse IlluminationConditions for Indoor Applications"**

# Hyperspectral Reconstruction from RGB Images for Vein Visualization
Hyperspectral imaging provides rich information across many wavelengths of the captured scene, which is useful for many potential applications such as food quality inspection, medical diagnosis, material identification, artwork authentication, and crime scene analysis. However, hyperspectral imaging has not been widely deployed for such indoor applications. In this work, we address one of the main challenges stifling this wide adoption, which is the strict illumination requirements for hyperspectral cameras. Hyperspectral cameras require a light source that radiates power across a wide range of the electromagnetic spectrum. Such light sources are expensive to setup and operate, and in some cases, they are not possible to use because they could damage important objects in the scene. We propose a data-driven method that enables indoor hyperspectral imaging using cost-effective and widely available lighting sources such as LED and fluorescent. These common sources, however, introduce significant noise in the hyperspectral bands in the invisible range, which are the most important for the applications. Our proposed method restores the damaged bands using a carefully-designed supervised deep-learning model. We conduct an extensive experimental study to analyze the performance of the proposed method using real hyperspectral datasets that we have collected. Our results show that the proposed method outperforms the state-of-the-art across all considered objective and subjective metrics, and it produces hyperspectral bands that are close to the ground truth bands captured under ideal illumination conditions.


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
cd HS_In_Diverse_Illuminations
```
- Install [PyTorch](http://pytorch.org) and other dependencies.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.



### Train  


#### Train Dataset preparation:
- The model requires 50x50x25 patches sampled in `.h5` from augmented dataset for the training process.
- Original Augmented dataset in `.mat` format for the Fruit Processing and Material Identification is available at [Fruit](https://drive.google.com/drive/folders/1BI6J3aJiuqpXMFlNwYt3O0JLP3PHW4zD?usp=sharing) and [Material](https://drive.google.com/drive/folders/1LBvEqoJuQ3o9ryulqWbktEmI3K-g-K_1?usp=sharing)
- `save_h5_data_fruit.m` and `save_h5_data_material.m` are used for sampling and constructing `.h5` files. 
- Training and Validation `.h5` files for the both Fruit Processing and Material Identification categories are available at: [Fruit Processing Training](https://drive.google.com/file/d/1qQGmerp7RU6igRSg7gUWX62EvTj1YYsS/view?usp=sharing), [Fruit Processing Validation](https://drive.google.com/file/d/1EvY3f-Rbm2FYMmw7SWA30pbO4WyTWXqz/view?usp=sharing), [Material Identification Training](https://drive.google.com/file/d/1fhotXS85J7Bt1oH8AHxa4zNt9fon1wJt/view?usp=sharing), and [Material Identification Validation](https://drive.google.com/file/d/1_hZJZIYA2yI0v2WRkpIFpur6ae8ldCup/view?usp=sharing) sets.
- Move these downloaded `.h5` files to `./train/Data` folder


#### Train the model:
The training and testing codes are present in `./train/` folder. The model architecture is present in `resblock.py` file.
- Train a model for the Fruit Processing Application:
```bash
#!./train/train_fruit.py
python train_fruit.py
```


- Train a model for the Material Identification Application:
```bash
#!./train/train_material.py
python train_material.py
```

- The trained models will be stored in `./train/models/` folder with log files. 


### Test
#### Test Dataset preparation:
- Test Dataset for Fruit Processing and Material Identification Applications is available at: [Test Data](https://drive.google.com/file/d/1a3R77JJvedsuCH8KoR_m5H_BOaw62fA1/view?usp=sharing)

- Extract `Data.zip` file and transfer `Data` folder into `./test/`. 

#### Test the models:
- The pre-trained models for both applications are present in `./test/models/`.
- Test pre-trained model for fruit processing application:
```bash
#!./test/evaluate_model_fruit.py
python evaluate_model_fruit.py
```
- Test pre-trained model for material identification application:
```bash
#!./test/evaluate_model_material.py
python evaluate_model_material.py
```

- Results are available in `./test/Data/Fruit/test_results` and `./test/Data/Material/test_results` in `.mat` format. You can analyze them using matlab.


