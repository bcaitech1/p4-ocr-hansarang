import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from detect import predict
from inference import main
	
st.title('OCR_prototype')
	
DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
              'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
	
# @st.cache
# def load_data(nrows):
# 	data = pd.read_csv(DATA_URL, nrows=nrows)
# 	lowercase = lambda x: str(x).lower()
# 	data.rename(lowercase, axis='columns', inplace=True)
# 	data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
# 	return data
#
#
# def file_selector(folder_path='.'):
#     filenames = os.listdir(folder_path)
#     selected_filename = st.selectbox('Select a file', filenames)
#     return os.path.join(folder_path, selected_filename)


st.title("Upload + Classification Example")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    # label = predict(uploaded_file)
    label = predict(image)
    # st.write('%s (%.2f%%)' % (label[1], label[2]*100))
    st.write(label)


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
