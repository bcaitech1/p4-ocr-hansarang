# Pstage 4 – OCR – 한사랑개발회

<img src="C:\Users\TGJ\Desktop\BoostCamp_AI\P_stage\4_stage\깃문서 작성\수식 인식 데모1.png" alt="(임시)수식인식기 데모" style="zoom:75%;" />

수식인식기는 OCR(Optical Character Recognition) task의 일환으로써 수식 이미지 latex 포맷의 텍스트로 변환하는 task입니다. 수식 인식의 경우, 기존의 OCR과 달리 multi line recogniton을 필요로 한다는 점에서 기존의 single line recognition OCR task와 차별점을 가집니다.

(우리 수식인식기 특징 정리)::저희 수식인식기 task는 SATRN model을 기본 틀로 설정하여 수식 이미지 학습을 진행했습니다. Multi line recognition 문제 이외에도 다른 문제를 해결하기 위해 기존 model에서 몇가지 개선 사항을 진행했습니다. 추가적으로 python flask를 이용해 개발한 간단한 데모를 만듦으로써 저희가 만든 모델을 서비스로 출시할 수 있는 가능성을 제시하고자 합니다.

Wrap-up Report에 문제 인식, EDA, 모델 설계, 실험 관리 및 검증 전략 등 저희가 다룬 기술의 흐름과 고민의 흔적을 기록했습니다. 

 

# Table of Contents

* [Demo](#Demo)
* [Getting started](#Getting-started)
  * [Dependencies](#Dependencies)
  * [Installation](#Installatio)
  * [Train](#Train)
  * [Inference](#Inference)

* [File Structure](#File-Structure)
  * [Baseline code](#Baseline-code)
  * [Input](#Input)
* [Yaml file example](#Yaml-file-example)
* [Overview](Overview)
* [Contributors](Contributors)
* [Reference](Reference)
* [Licence](Licence)

## Demo

[[Click here!]Demo link](http://35.74.99.158:8501/)
아래의 링크에서 Demo의 자세한 사항을 확인하실 수 있습니다.

* [streamlit_test](https://github.com/bcaitech1/p4-ocr-hansarang/tree/main/streamlit_test)

![demo_example](https://user-images.githubusercontent.com/43458619/122248828-54d48f00-cf03-11eb-81cd-d3ba2aa3dd2a.png)



## Getting started

### Dependencies

(requirements 파일 참조)

- scikit_image==0.14.1
- opencv_python==3.4.4.19
- tqdm==4.28.1
- torch==1.4.0
- scipy==1.2.0
- numpy==1.15.4
- torchvision==0.2.1
- Pillow==8.1.1
- tensorboardX==1.5
- editdistance==0.5.3

(추가 설치 파일 참조 -> 최종 코드에 맞게 수정)

- opencv-python==3.4.4.19
- adamp==0.3.0 
- madgrad==1.1 
- timm==0.4.9
- wandb==0.10.31

### Installation

```
$ git clone https://github.com/bcaitech1/p4-ocr-hansarang.git
$ cd code <-- (수정 필요)
```

### Train

```
$ python train.py --config_file ./configs/SATRN.yaml
```

### Inference

```
$ python train.py --config_file ./log/satrn/checkpoints/0050.pth
```



## File Structure

### Baseline code

```
code
├── README.md
├── __pycache__
│   ├── checkpoint.cpython-37.pyc
│   ├── dataset.cpython-37.pyc
│   ├── flags.cpython-37.pyc
│   ├── metrics.cpython-37.pyc
│   ├── scheduler.cpython-37.pyc
│   ├── train.cpython-37.pyc
│   └── utils.cpython-37.pyc
├── checkpoint.py
├── configs
│   ├── Attention.yaml
│   └── SATRN.yaml
├── data_tools
│   ├── extract_tokens.py
│   ├── parse_upstage.py
│   └── train_test_split.py
├── dataset.py
├── download.sh
├── flags.py
├── inference.py
├── log
├── metrics.py
├── networks
│   ├── Attention.py
│   ├── Mish.py
│   ├── SATRN.py
│   ├── SATRN_Effnet.py
│   ├── SATRN_Mish.py
│   ├── Swish.py
│   ├── __pycache__
│   │   ├── Attention.cpython-37.pyc
│   │   ├── SATRN.cpython-37.pyc
│   │   └── SATRN_Effnet.cpython-37.pyc
│   └── spatial_transformation.py
├── requirements.txt
├── scheduler.py
├── submission.txt
├── submit
│   └── output.csv
├── train.py
├── utils.py
```



### Input

```
input
└── data
    ├── eval_dataset
    │   ├── images
    │   └── input.txt
    └── train_dataset
        ├── gt.txt
        ├── images
        ├── level.txt
        ├── source.txt
        └── tokens.txt
```



## Yaml file example

모델 학습에 사용할 때 사용하는 config 파일(ex. SATRN.yaml) 예시입니다.

예시 config 파일들은 [code/configs]()(경로 수정 해야 함)에 존재하니, 참고하여 config 파일을 작성해 자유롭게 학습을 진행하실 수 있습니다.

```yaml
network: SATRN
input_size:
  height: 100
  width: 400
SATRN:
  encoder:
    hidden_dim: 300
    filter_dim: 600
    layer_num: 6
    head_num: 8
  decoder:
    src_dim: 300
    hidden_dim: 128
    filter_dim: 512
    layer_num: 3
    head_num: 8
Attention:
  src_dim: 512
  hidden_dim: 128
  embedding_dim: 128
  layer_num: 1
  cell_type: "LSTM"
checkpoint: ""
prefix: "./log/satrn"

data:
  train:
    #- "/opt/ml/input/data/train_dataset/gt.txt"
    - "../../train_dataset/gt.txt"
  test:
    - "../../train_dataset/tokens.txt"
  token_paths:
    #- "/opt/ml/input/data/train_dataset/tokens.txt"  # 241 tokens
    -
  dataset_proportions:  # proportion of data to take from train (not test)
    - 1.0
  random_split: True # if True, random split from train files
  test_proportions: 0.2 # only if random_split is True
  crop: True
  rgb: 1    # 3 for color, 1 for greyscale
  
batch_size: 25
num_workers: 8
num_epochs: 50
print_epochs: 1
dropout_rate: 0.1
teacher_forcing_ratio: 0.8
max_grad_norm: 2.0
seed: 1234
optimizer:
  optimizer: 'NAdam' # Adam, Adadelta
  lr: 5e-4 # 1e-4
  weight_decay: 1e-4
  is_cycle: True
```



## Overview

(그림 작성 필요)



## Contributors

[박재우(JJayy)](https://github.com/JJayy) | [송광원(remaindere)]() | [신찬엽(chanyub)](https://github.com/chanyub) | [조원(jo-member)](https://github.com/jo-member) | [탁금지(Atica57)](https://github.com/Atica57) | [허재섭(shjas94)](https://github.com/shjas94)



## Reference

* [On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention](https://arxiv.org/abs/1910.04396)



## License

 본 프로젝트는 아래의 license를 따릅니다.

```
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF ERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

