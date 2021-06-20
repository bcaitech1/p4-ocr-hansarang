# 한사랑 개발회 수식 OCR 모델 프로토타입

## ScreenShot


## Requirement
상위 requirement + Streamlit

## 사용방법
1. ```git clone https://github.com/bcaitech1/p4-ocr-hansarang```
2. ```pip install streamlit``` 혹은 ```conda install streamlit``` 으로 streamlit을 설치
3. [pth파일](https://drive.google.com/file/d/16ObA_IKs79ZuKhNgsay-bTHrs-tfEn9y/view?usp=sharing) 을 log/satrn/checkpoints에 0050.pth 파일을 저장
4. 설정을 마친 후에 ```streamlit run demo.py``` 를 입력 후 로컬에서 웹사이트를 확인

## 작동 방식
1. demo.py에서 이미지를 받아 ocr_core에 parsing 해줍니다.
2. ocr_core는 받아온 이미지를 전처리 후 모델에 넘겨 주어 하여 추론 값을 얻어냅니다.
3. 얻어낸 추론 값을 return 해 준 것을 demo.py가 가져옵니다.
4. 결과를 화면에 출력해 줍니다.

### 해당 데모를 위한 구성 요소들은 demo/Demo.zip에 따로 저장해두었습니다!