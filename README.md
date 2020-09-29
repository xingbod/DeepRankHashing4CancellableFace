# [IoMFace-tf2/1](https://github.com/charlesLucky/IoMArcFace)


****

## Contents
* [Installation](#Installation)
* [Data Preparing](#Data-Preparing)
* [Training and Testing](#Training-and-Testing)
* [Using InsightFace pre_build model](#Using-InsightFace-pre_build-model)
* [References](#References)

## Installation
Create a new python virtual environment by [Anaconda](https://www.anaconda.com/) or just use pip in your python environment and then clone this repository as following.

### Clone this repo
```bash
git clone https://github.com/charlesLucky/IoMArcFace.git
cd IoMArcFace
```

### Conda
```bash
conda env create -f environment.yml
conda activate deepIoM-tf2
```

If you use tensorflow 1.15.0, kindly use:

```bash
conda env create -f environment_tf1.yml
conda activate deepIoM-tf1
```

### Pip

```bash
pip install -r requirements.txt
```

If you use tensorflow 1.15.0, kindly use:

```bash
pip install -r requirements_tf1.txt
```
****

If encounter any probelm, you may need to use:

```bash
conda install tensorflow-gpu==1.15.0
```

## Try the model

If you want to try first, you can run the run_me.py file, and it will firstly load the model, and then read images of two people. The hash code will be generated and the distance will also be computed. The weight file in section [Using InsightFace pre_build model](#Using InsightFace pre_build model) is needed before you use the following example. 


You can finally get the distance between same person and different person:

The command is like below:

```bash
IoMArcFace$  python run_me.py --cfg_path .configs/config_15/cfg15_allloss_res100_512x8.yaml
Num GPUs Available:  1
1 Physical GPUs, 1 Logical GPUs
[*] Warning!!!! Cannot find ckpt from None, random weight of IoM layer will be used.
Model: "IoMface_model"
________________________________________________________________________________
Layer (type)                        Output Shape                    Param #     
================================================================================
input_image (InputLayer)            [(None, 112, 112, 3)]           0           
________________________________________________________________________________
arcface_model (Model)               (None, 512)                     70899904    
________________________________________________________________________________
perm_layer (PermLayer)              (None, 512)                     0           
________________________________________________________________________________
IoMProjection (Dense)               (None, 4096)                    2101248     
________________________________________________________________________________
IoMHead (Model)                     (None, 512)                     0           
================================================================================
Total params: 73,001,152
Trainable params: 72,929,472
Non-trainable params: 71,680
________________________________________________________________________________
dist_positive: 318.0, dist_negative: 443.0
```

The model consists of a feature extractor network, a permutation layer, plus a IoM projection layer.


## Data Preparing
All datasets used in this repository can be found from [face.evoLVe.PyTorch's Data-Zoo](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch#Data-Zoo).

Note:

- Both training and testing dataset are "Align_112x112" version.

### Training Dataset
To train the ArcFace, MS-Celeb-1M need to be downloaded first:

Download [MS-Celeb-1M](https://drive.google.com/file/d/1X202mvYe5tiXFhOx82z4rPiPogXD435i/view?usp=sharing) datasets, then extract and convert them to tfrecord as traning data as following.

```bash
# Binary Image: convert really slow, but loading faster when traning.
python data/convert_train_binary_tfrecord.py --dataset_path="/path/to/ms1m_align_112/imgs" --output_path="./data/ms1m_bin.tfrecord"

# Online Image Loading: convert really fast, but loading slower when training.
python data/convert_train_tfrecord.py --dataset_path="/path/to/ms1m_align_112/imgs" --output_path="./data/ms1m.tfrecord"
```

Note:
- You can run `python ./dataset_checker.py` to check if the dataloader work.

To train the deep IoM, use the dataset directory directly, no need to convert images into binary due to the use of triple sampling.

### Testing Dataset

Download [LFW](https://drive.google.com/file/d/1WO5Meh_yAau00Gm2Rz2Pc0SRldLQYigT/view?usp=sharing), [Aged30](https://drive.google.com/file/d/1AoZrZfym5ZhdTyKSxD0qxa7Xrp2Q1ftp/view?usp=sharing) and [CFP-FP](https://drive.google.com/file/d/1-sDn79lTegXRNhFuRnIRsgdU88cBfW6V/view?usp=sharing) datasets, then extract them to `/your/path/to/test_dataset`. These testing data are already binary files, so it's not necessary to do any preprocessing. The directory structure should be like bellow.
```
/your/path/to/test_dataset/
    -> lfw_align_112/lfw
        -> data/
        -> meta/
        -> ...
    -> agedb_align_112/agedb_30
        -> ...
    -> cfp_align_112/cfp_fp
        -> ...
```

****

## Training and Testing
### To train ArcFace model
You can modify your own dataset path or other settings of model in [./configs/*.yaml]() for training and testing, which like below.

```python
# general (shared both in training and testing)
batch_size: 128
input_size: 112
embd_shape: 512
sub_name: 'arc_res50'
backbone_type: 'ResNet50' # or 'MobileNetV2'
head_type: ArcHead # or 'NormHead': FC to targets.
is_ccrop: False # central-cropping or not

# train
train_dataset: './data/ms1m_bin.tfrecord' # or './data/ms1m.tfrecord'
binary_img: True # False if dataset is online decoding
num_classes: 85742
num_samples: 5822653
epochs: 5
base_lr: 0.01
w_decay: !!float 5e-4
save_steps: 1000

# test
test_dataset: '/your/path/to/test_dataset'
```

Note:
- The `sub_name` is the name of outputs directory used in checkpoints and logs folder. (make sure of setting it unique to other models)
- The `head_type` is used to choose [ArcFace](https://arxiv.org/abs/1801.07698) head or normal fully connected layer head for classification in training. (see more detail in [./modules/models.py](https://github.com/peteryuX/arcface-tf2/blob/master/modules/models.py#L90-L94))
- The `is_ccrop` means doing central-cropping on both trainging and testing data or not.
- The `binary_img` is used to choose the type of training data, which should be according to the data type you created in the [Data-Preparing](#Data-Preparing).

### To train deep IoM in the second stage

The model (checkpoints file) of the first stage shall be stored under [checkpoints/arc_res50/*]()

You can modify settings of model in [./config_*/*.yaml]() for training and testing, which like below.

```python
# general
samples_per_class: 4
classes_per_batch: 50
batch_size: 240
eval_batch_size: 200
input_size: 112
# embd_shape is for the Resnet, backbone
embd_shape: 512
```
```python
sub_name: 'cfg9_1layer_arc_all_T300_256x32_0'
backbone_type: 'ResNet50' # 'ResNet50', 'MobileNetV2'
head_type: IoMHead # 'ArcHead', 'NormHead'
bin_lut_loss: 'tanh'
hidden_layer_remark: '1'
#T: 300
code_balance_loss: True
quanti: True
# train /media/xingbo/Storage/facedata/vgg_mtcnnpy_160 ./data/split /media/xingbo/Storage/facedata/ms1m_align_112/imgs
train_dataset: '/home/datascience/xingbo/ms1m_align_112/imgs'
#train_dataset: '/media/xingbo/Storage/facedata/vgg_mtcnnpy_160'
img_ext: 'jpg'
dataset_ext: 'ms'
# for metric learning, we have 1. triplet_semi batch_hard_triplet 2. batch_all_triplet_loss 3. batch_all_arc_triplet_loss batch_hard_arc_triplet_loss
loss_fun: 'batch_hard_arc_triplet_loss'
triplet_margin: 0
binary_img: False
num_classes: 85742
# I generate 1 milion triplets
num_samples: 3000000
epochs: 50
base_lr: 0.001
w_decay: !!float 5e-4
save_steps: 10000
q: 32
# m, the projection length would be m x q
m: 256
# test
test_dataset: './data/test_dataset'
test_dataset_ytf: './data/test_dataset/aligned_images_DB_YTF/'
test_dataset_fs: './data/test_dataset/facescrub_images_112x112/'
```
Note:
- The `sub_name` is the name of outputs directory used in checkpoints and logs folder. (make sure of setting it unique to other models)
- The `head_type` is used to choose [ArcFace](https://arxiv.org/abs/1801.07698) head or normal fully connected layer head for classification in training. (see more detail in [./modules/models.py](https://github.com/peteryuX/arcface-tf2/blob/master/modules/models.py#L90-L94))
- The `bin_lut_loss` is the name of binarization look up table (LUT) loss used in training. (tanh,sigmoid, or none)
- The `hidden_layer_remark` means how many hidden layers used. (possible value: 1,2,3,T1,T2)
- The `code_balance_loss` means using code balance loss on trainging or not.
- The `quanti` means using quantization loss on trainging or not.
- The `loss_fun` is the name of training loss used in traning. (possible value:batch_hard_triplet,batch_all_triplet_loss,batch_all_arc_triplet_loss,batch_hard_arc_triplet_loss,semihard_triplet_loss )
- The `triplet_margin` is the name of outputs directory used in checkpoints and logs folder. (make sure of setting it unique to other models)
- The `q` q in IoM
- The `m` m in IoM

### Training

Stage 1: Here have two modes for training the arcface your model, which should be perform the same results at the end.
```bash
# traning with tf.GradientTape(), great for debugging.
python train.py --mode="eager_tf" --cfg_path="./configs/config_arc/arc_res50.yaml"

# training with model.fit().
python train.py --mode="fit" --cfg_path="./configs/config_arc/arc_res50.yaml"
```
Stage 2: Training the deep IoM:

```bash
# traning with tf.GradientTape(),For deep IoM, can only train by eager_tf
nohup python -u train_twostage_tripletloss_online.py --cfg_path ./configs/config_10/iom_res50_twostage_1layer_hard_arcloss_256x8_0.yaml >1layer_hard_arcloss_256x8_0.log & 


### Testing the performance of deep IoM

```bash
python  test_twostage_iom.py --cfg_path ./configs/config_10/iom_res50_twostage_1layer_hard_arcloss_256x8_0.yaml 
```

# Using-InsightFace-pre_build-model
In this work, we also try to adopt the original pre-build model by InsightFace team. However, their original model is trained on Mxnet, which is not fit tensorflow directly. Hence we perform the model conversion firstly to generate a tensorflow model. 

We adopted their ResNet100 model, the original performance is:

<table><thead><tr><th>Method</th><th>LFW(%)</th><th>CFP-FP(%)</th><th>AgeDB-30(%)</th><th>MegaFace(%)</th></tr></thead><tbody><tr><td>Ours</td><td>99.77</td><td>98.27</td><td>98.28</td><td>98.47</td></tr></tbody></table>

While after the model conversion, the generated TF2 model performance is:

<table><thead><tr><th></th><th>LFW</th><th>AgeDB30</th><th>CFP - FP</th></tr></thead><tbody><tr><td>Accuracy</td><td>0.9960</td><td>0.9752</td><td>0.9643</td></tr><tr><td>EER</td><td>0.0040</td><td>0.0305</td><td>0.0387</td></tr><tr><td>AUC</td><td>0.9987</td><td>0.9900</td><td>0.9877</td></tr><tr><td>Threshold</td><td>0.7340</td><td>0.7710</td><td>0.8320</td></tr></tbody></table>

There is a slightly accuracy drop, but it is still better than our own trained model.

To use this pre-build model, just set the **backbone_type** in the config file as Insight_ResNet100:

```
batch_size: 16
eval_batch_size: 16
input_size: 112
embd_shape: 512
sub_name: 'arc_Insight_ResNet100'
backbone_type: 'Insight_ResNet100' # 'ResNet50', 'MobileNetV2'
head_type: ArcHead # 'ArcHead', 'NormHead'
is_ccrop: False # central-cropping or not
```

Please note that the weight file is required, it is stored in `pre_models/resnet100/resnet100.npy`

The weight file and other related files can be downloaded from [this link](https://drive.google.com/file/d/1aOy12NnkEBmzLa9atQQCAlKiRO8zck49/view?usp=sharing).

## References

Thanks for these source codes porviding me with knowledges to complete this repository.

- https://github.com/peteryuX/arcface-tf2
    - ArcFace (Additive Angular Margin Loss for Deep Face Recognition, published in CVPR 2019) implemented in Tensorflow 2.0+. This is an unofficial implementation.
- https://github.com/deepinsight/insightface (Official)
    - Face Analysis Project on MXNet http://insightface.ai
- https://github.com/zzh8829/yolov3-tf2
    - YoloV3 Implemented in TensorFlow 2.0
- https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
    - face.evoLVe: High-Performance Face Recognition Library based on PyTorch
- https://github.com/luckycallor/InsightFace-tensorflow
    - Tensoflow implementation of InsightFace (ArcFace: Additive Angular Margin Loss for Deep Face Recognition).
- https://github.com/dmonterom/face_recognition_TF2
    - Training a face Recognizer using ResNet50 + ArcFace in TensorFlow 2.0
