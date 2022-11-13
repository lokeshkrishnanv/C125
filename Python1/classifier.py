import numpy as np
import pandas as pd
import PIL.ImageOps
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = fetch_openml("mnist_784", version=1, return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)
X_train_scale = X_train/255
X_test_scale = X_test/255

clf = LogisticRegression(solver="saga", multi_class = "multinomial").fit(X_train_scale, y_train)

def get_prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    return test_pred[0]