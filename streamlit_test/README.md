# 한사랑 개발회 수식 OCR 모델 프로토타입

![screensh]('/show_img.png')

## 사용방법
1. git clone
2. streamlit_test.py, detect.py, inference.py에서 #must change라고 붙여둔 곳을 자기 로컬 환경에 맞게 변경해줍니다.
3. 학습된 pth파일을 log/attention_50/checkpoints에 있는 0050.pth 파일 대신 덮어써주면 그 모델로 inference가 가능합니다.
4. 설정을 마친 후에 streamlit run streamlit_test.py 사용하시면 로컬에서 웹사이트를 확인할 수 있습니다.

## 작동 방식
1. streamlit_test.py에서 이미지를 받아옴
2. 받아온 이미지를 data/images 폴더 내에 저장
3. inference.py에서 저장된 파일을 미리 학습된 모델로 예측함
4. 그 결과를 화면에 띄워줌

## 해결과제
1. 서버에 사진을 저장하지 않고도 결과 도출이 가능하도록 수정할 것
2. 결과로 나오는 Latex를 보기좋게 수정할 것
3. 모델 inference 성능 향상 필요
4. 배포하기

## 추가 예정 서비스
1. 웹캠과의 연동
2. 배포 후 핸드폰에서도 사용가능하도록 수정

## 의문점
1. 지금처럼 서버에 사진을 저장하면 여러명 동시접속할때 결과가 잘못 나오지 않나요? ㅠㅠ
2. 서버에 저장 안하고 이걸 어떻게 하죠? ㅠㅠ




-----이 밑 내용은 나중에 수정할때 참고하려고 남겨둔 내용(일단 무시)-----





## Requirements

- Python 3
- [PyTorch][pytorch]

All dependencies can be installed with PIP.

```sh
pip install tensorboardX tqdm pyyaml psutil
```

현재 검증된 GPU 개발환경으로는
- `Pytorch 1.0.0 (CUDA 10.1)`
- `Pytorch 1.4.0 (CUDA 10.0)`
- `Pytorch 1.7.1 (CUDA 11.0)`


## Supported Models

- [CRNN][arxiv-zhang18]
- [SATRN](https://github.com/clovaai/SATRN)


## Supported Data
- [Aida][Aida] (synthetic handwritten)
- [CROHME][CROHME] (online handwritten)
- [IM2LATEX][IM2LATEX] (pdf, synthetic handwritten)
- [Upstage][Upstage] (print, handwritten)


모든 데이터는 팀 저장소에서 train-ready 포맷으로 다운 가능하다.
```
[dataset]/
├── gt.txt
├── tokens.txt
└── images/
    ├── *.jpg
    ├── ...     
    └── *.jpg
```


## Usage

### Training

```sh
python train.py
```


### Evaluation

```sh
python evaluate.py
```

[arxiv-zhang18]: https://arxiv.org/pdf/1801.03530.pdf
[CROHME]: https://www.isical.ac.in/~crohme/
[Aida]: https://www.kaggle.com/aidapearson/ocr-data
[Upstage]: https://www.upstage.ai/
[IM2LATEX]: http://lstm.seas.harvard.edu/latex/
[pytorch]: https://pytorch.org/
