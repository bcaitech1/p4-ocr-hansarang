from inference import main
import os,sys
import cv2
import numpy as np
import pandas as pd
def predict(image1):
    image1 = np.array(image1)
    cv2.imwrite('/Users/sinchan-yeob/streamlit_test/data/images/train_00000.jpg',image1) # must change
    test = "python inference.py"
    os.system(test)
    f = open('/Users/sinchan-yeob/streamlit_test/submit/output.csv', 'r') # must change
    test = f.read()
    file_name = test.split('\t')[0]
    latex_info = test.split('\t')[1:]
    return latex_info