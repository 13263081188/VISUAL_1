# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import requests
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import  pandas as pd
from PIL import Image
from io import BytesIO
from torchvision import models
from torchvision.transforms.functional import resize, to_tensor, normalize, to_pil_image
from torchcam import cams
from torchcam.utils import overlay_mask
CAM_METHODS = ["CAM", "GradCAM", "GradCAMpp", "SmoothGradCAMpp", "ScoreCAM", "SSCAM", "ISCAM", "XGradCAM"]
TV_MODELS = ["resnet18", "resnet50", "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"]
LABEL_MAP = requests.get(
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
).json()

# @st.cache
def main():
    
    # Wide mode
    st.set_page_config(layout="wide")
    # Designing the interface
    st.title("ZXLariation auto-encoder")
    # For newline
    st.write('\n')
    test = st.beta_columns(3)
    zz = test[0].form("input image")
    # zz.form_submit_button("dont'touch")
    #cam_ for i in range(1000)
    cols = [st.form(str(i)) for i in range(4)]
    # cols[0].write("Input image")
    # zz.form_submit_button("DONT'touch me")
    st.write('\n')
    for i in range(4):
        cols[i].write(CAM_METHODS[i])
        cols[i].form_submit_button("COMPUTE " + CAM_METHODS[i])
        # x,y,z = cols[i + 1].beta_columns(3)
        # x.write("1")
        # y.write("2")
        # z.write("3")

    st.title('Uber pickups in NYC')
    DATE_COLUMN = 'date/time'
    DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
                'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

    @st.cache
    def load_data(nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        return data

    data_load_state = st.text('Loading data...')
    data = load_data(10000)
    data_load_state.text("Done! (using st.cache)")

    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)

    st.subheader('Number of pickups by hour')
    hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
    st.bar_chart(hist_values)

    # Some number in the range 0-23
    hour_to_filter = st.slider('hour', 0, 23, 17)
    filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

    st.subheader('Map of all pickups at %s:00' % hour_to_filter)
    st.map(filtered_data)





































    # Sidebar
    # File selection
    st.sidebar.title("Input selection")
    # Disabling warning
    st.set_option('deprecation.showfileUploaderEncoding', False)
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader("Upload files", type=['png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        # zz.write("Aa Aa Aa")
        # st.balloons()
        img = Image.open(BytesIO(uploaded_file.read()), mode='r').convert('RGB')
        zz.image(img,use_column_width=True)
    if zz.form_submit_button("DONT'touch me"):
        # if uploaded_file is not None:
        # st.balloons()
        pass
    #Model selection
    st.sidebar.title("Setup")
    tv_model = st.sidebar.selectbox("Classification model", TV_MODELS)
    default_layer = ""
    if tv_model is not None:
        with st.spinner('Loading model...'):
            model = models.__dict__[tv_model](pretrained=True).eval()
        default_layer = cams.utils.locate_candidate_layer(model, (3, 224, 224))
    l_num = list(range(32))
    latent_num = st.sidebar.selectbox("latent_num", l_num)
    vae_model = st.sidebar.selectbox("VAE model", ["vanila VAE"])
    # if vae_model is not None and latent_num is not None:
    #     with st.spinner('Loading model...'):
    #         model = models.__dict__[tv_model](pretrained=True).eval()
    default_layer = cams.utils.locate_candidate_layer(model, (3, 224, 224))
    default_layer = ""
    target_layer = st.sidebar.text_input("Target layer", default_layer)
    cam_method = st.sidebar.selectbox("CAM method", CAM_METHODS)
    # st.write(cam_method)
    if cam_method is not None:
        cam_extractor = cams.__dict__[cam_method](
            model,
            target_layer=target_layer if len(target_layer) > 0 else None
    )
    # st.write(cam_method)
    class_choices = [f"{idx + 1} - {class_name}" for idx, class_name in enumerate(LABEL_MAP)]
    class_selection = st.sidebar.selectbox("Class selection", ["Predicted class (argmax)"] + class_choices)
    # cols = [st.form(str(i)) for i in range(4)]
    # cols[0].write("Input image")
    # st.write('\n')
    for i in range(4):
        # cols[i + 1].form_submit_button("COMPUTE " + CAM_METHODS[i])
    # for i in range(1,4):
        if cols[i].form_submit_button("COMPUTE " + CAM_METHODS[i]):
            st.balloons()
            if uploaded_file is None:
                st.sidebar.error("Please upload an image first")
            else:
                st.balloons()
                with st.spinner('Analyzing...'):
                    # Preprocess image
                    img_tensor = normalize(to_tensor(resize(img, (224, 224))), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    # Forward the image to the model
                    out = model(img_tensor.unsqueeze(0))
                    # Select the target class
                    if class_selection == "Predicted class (argmax)":
                        class_idx = out.squeeze(0).argmax().item()
                    else:
                        class_idx = LABEL_MAP.index(class_selection.rpartition(" - ")[-1])
                    # Retrieve the CAM
                    activation_map = cam_extractor(class_idx, out)
                    x, y, z = cols[i].beta_columns(3)
                    # Plot the raw heatmap
                    fig, ax = plt.subplots()
                    ax.imshow(activation_map.numpy())
                    ax.axis('off')
                    # cols_1,cols_2,cols_3 = cols[i].beta_columns(3)
                    x.imagine(img,use_column_width=True)
                    y.imagine(img,use_column_width=True)
                    # cols_1.write('1')
                    # cols_2.write("1")
                    # Overlayed CAM
                    fig, ax = plt.subplots()
                    result = overlay_mask(img, to_pil_image(activation_map, mode='F'), alpha=0.5)
                    ax.imshow(result)
                    ax.axis('off')
                    # cols_3.write("1")
                    # cols_2.pyplot(fig)
                    z.imagine(img,use_column_width=True)

if __name__ == '__main__':
    main()
