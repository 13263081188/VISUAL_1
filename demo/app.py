# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import requests
import streamlit as st
import matplotlib.pyplot as plt
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
def main():
    # Wide mode
    st.set_page_config(layout="wide")
    # Designing the interface
    st.title("variation auto-encoder")
    # For newline
    st.write('\n')
    #cam_ for i in range(1000)
    cols = st.beta_columns(4)
    cols[0].header("Input image")
    st.write('\n')
    for i in range(3):
        cols[i+1].header(CAM_METHODS[i])
    # Sidebar
    # File selection
    st.sidebar.title("Input selection")
    # Disabling warning
    st.set_option('deprecation.showfileUploaderEncoding', False)
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader("Upload files", type=['png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        img = Image.open(BytesIO(uploaded_file.read()), mode='r').convert('RGB')
        cols[0].image(img, use_column_width=True)
    # Model selection
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
    if st.sidebar.button("ComputeCAM"):
        if uploaded_file is None:
            st.sidebar.error("Please upload an image first")
        else:
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
                # Plot the raw heatmap
                fig, ax = plt.subplots()
                ax.imshow(activation_map.numpy())
                ax.axis('off')
                cols[1].pyplot(fig)
                # Overlayed CAM
                fig, ax = plt.subplots()
                result = overlay_mask(img, to_pil_image(activation_map, mode='F'), alpha=0.5)
                ax.imshow(result)
                ax.axis('off')
                cols[-1].pyplot(fig)
                cols[-2].pyplot(fig)
                # cols[-3].pyplot(fig)

if __name__ == '__main__':
    main()