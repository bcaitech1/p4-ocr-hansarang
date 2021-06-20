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
