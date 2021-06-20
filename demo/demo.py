import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
from ocr_core import ocr_core
from matplotlib import pyplot as plt
from inference import main
from streamlit_drawable_canvas import st_canvas

st.title('한사랑개발회 OCR 프로토타입')

st.title("수식인식기")
st.subheader("파일 업로드")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
###drawer###
mode = st.checkbox("Draw (or Delete)?", True)
st.subheader("직접 그리기")
canvas_result = st_canvas(
    fill_color='#FFFFFF',
    stroke_width=8,
    stroke_color='#000000',
    background_color='#FFFFFF',
    width=698,
    height=200,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (400, 100))
    rescaled = cv2.resize(img, (400, 100), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test_x = Image.fromarray(test_x)
    gif_runner = st.image('rocket.gif')
    val = ocr_core(test_x)
    gif_runner.empty()
    st.latex(val)
    st.write(f'result: {val}')
###drawer###

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    # label = predict(uploaded_file)
    gif_runner = st.image('rocket.gif')
    label = ocr_core(image)
    gif_runner.empty()
    # st.markdown('$' + label + '$')
    st.latex(label)
    st.text(label)
    # label = predict(image)
    # st.markdown('$'+label+'$')
    # latex_to_img(label)


# data_load_state = st.text('Loading data...')
# data = load_data(10000)
# data_load_state.text("Done! (using st.cache)")
#
# if st.checkbox('Show raw data'):
# 	st.subheader('Raw data')
# 	st.write(data)

# filename = file_selector()
# st.write('You selected `%s`' % filename)

# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
# st.bar_chart(hist_values)
#
# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
#
# st.subheader('Map of all pickups at %s:00' % hour_to_filter)
# st.map(filtered_data)