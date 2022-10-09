from optical_flow_viz import viz_flow
from email.mime import image
import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from PIL import ImageColor, Image
import time
import os

import warnings
warnings.filterwarnings("ignore")


def main():

    # Introduction to Optical Flow

    st.title("Visualize Optical Flow ")
    st.subheader("Built with OpenCV and Streamlit")

    st.markdown('> #### Created by Dibaloke Chanda')

    st.markdown("""---""")

    st.markdown("> *Optical flow or optic flow is the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between an observer and a scene.*")

    st.text('Optical Flow Equation')

    st.latex(r'''
 \frac{\partial I}{\partial x} u+\frac{\partial I}{\partial y} v+\frac{\partial I}{\partial t}=0
    ''')

    with st.expander("See explanation"):
        st.image("of1.png")

        st.image("of2.png")

        st.text("Assuming constant pixel intensity")
        st.latex(r''' I(x, y, t)=I(x+\delta x, y+\delta y, t+\delta t) ''')

        st.text("Approximating with an Taylor series expansion:")

        st.latex(
            r''' I(x+\delta x, y+\delta y, t+\delta t)=I(x, y, t)+\frac{\partial I}{\partial x} \delta x+\frac{\partial I}{\partial y} \delta y+\frac{\partial I}{\partial t} \delta t''')

        st.latex(r''' I(x+\delta x, y+\delta y, t+\delta t)- I(x, y, t)= \frac{\partial I}{\partial x} \delta x+\frac{\partial I}{\partial y} \delta y+\frac{\partial I}{\partial t} \delta t
        ''')
        st.latex(r'''

        \frac{\partial I}{\partial x} \delta x+\frac{\partial I}{\partial y} \delta y+\frac{\partial I}{\partial t} \delta t =0

        ''')

        st.latex(r''' Dividing \ \  by  \ \ \delta t''')

        st.latex(
            r''' \frac{\partial I}{\partial x} u+\frac{\partial I}{\partial y} v+\frac{\partial I}{\partial t}=0 ''')

        st.latex(
            r''' where, \  \  u= \frac{dx}{dt} \ \  and \  v= \frac{dy}{dt} ''')
    st.markdown("""---""")

    st.subheader("Dense Optical Flow (Gunnar-Farneback Algorithm)")
    st.markdown("""---""")
    st.markdown("> **Webcam Implementation**")
    # Capture the Firstframe
    cap = cv2.VideoCapture(0)
    suc, prev = cap.read()
    prev_frame = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

 # Options for Parameter Selection

    options = st.radio(
        "What do you want to visualize ?",
        ('Flow Field + Image', 'Only Flow Field', 'Both'))

    col1, col2, col3 = st.columns(3)

    with st.expander('See explanation'):

        st.markdown(
            "+ **Pick a suitable color**: Choose a color depending on your background and lighting condition")
        st.markdown(
            "+ **Sparsity** : For High value more sparse field, for low value more dense field")

    with col1:
        color = st.color_picker(
            'Pick a suitable color', '#f90004', "visible")
    with col2:
        steps = st.slider('Sparsity', min_value=8,
                          max_value=64, value=16, step=1)
    with col3:
        dot_radius = st.slider('Dot Radius', min_value=1,
                               max_value=3, value=1, step=1)

    color_rgb = ImageColor.getcolor(color, "RGB")
    FRAME_WINDOW2 = st.image([])
    FRAME_WINDOW1 = st.image([])

    # st.subheader("Adjust Parameters")

    st.markdown("""---""")
    st.markdown("> Uploaded Video Implementation")

    uploaded_video = st.file_uploader("Choose video", type=["mp4"])

    if uploaded_video is not None:  # run only when user uploads video
        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read())  # save video to disk
        vidcap = cv2.VideoCapture(vid)  # load video from disk

        # Find video metadata
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        duration = frame_count/fps

        # cur_frame = 0
        # success = True

        st.video(uploaded_video, start_time=0)
        st.markdown("""---""")
        st.markdown(
            '> - Upload Low Resolution and Smaller clips for faster processing')

        st.markdown(
            '> - While Processing Video, Webcam implementation will be turned off. After processing completion, Webcam implementation will automatically turn on')
        st.markdown('> - Parameters can be changed on top')
        st.markdown(
            '> - Make sure you have ffmpeg installed and added to path. The following is code is being run to convert to "H264 codec" for browser compatibility')

        st.code(
            '''ffmpeg -i output.mp4 -vcodec libx264 output_streamlit.mp4''', language="bash")

        st.markdown("""---""")

        st.text('Total Number of Frames:'+str(frame_count))
        st.text('Video Duration:'+str(duration)+'second')
        st.text('FPS:'+str(fps))

        st.markdown("""---""")

        # my_bar = st.progress(0)

        # FRAME_WINDOW3 = st.image([])

        _, prev_video = vidcap.read()
        prev_frame_video = cv2.cvtColor(prev_video, cv2.COLOR_BGR2GRAY)
        h_video, w_video = prev_frame_video.shape[0], prev_frame_video.shape[1]
        frame_array = []
        frame_white_array = []

        with st.spinner('Processing Frames, Read the info above..'):
            for i in range(frame_count-1):
                _, frame_video = vidcap.read()  # get next frame from video
                frame_video = cv2.cvtColor(frame_video, cv2.COLOR_BGR2GRAY)
                flow_video = cv2.calcOpticalFlowFarneback(
                    prev_frame_video, frame_video, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                prev_frame_video = frame_video

                temp = viz_flow(frame_video, flow_video, step=steps,
                                color=color_rgb, dot_radius=dot_radius)

                frame_array.append(temp[1])
                frame_white_array.append(temp[0])

        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(
            *'mp4v'), fps, (w_video, h_video), True)

        out_white = cv2.VideoWriter('output_white.mp4', cv2.VideoWriter_fourcc(
            *'mp4v'), fps, (w_video, h_video), True)

        for i in range(frame_count-1):
            out.write(frame_array[i])
            out_white.write(frame_white_array[i])
        out.release()
        out_white.release()

        # st.text(type(skvideo.io.vread('output.mp4')))
        st.success('Done!')

        with st.spinner('Doing Codec conversion'):
            os.system(
                'ffmpeg -y -i output.mp4 -vcodec libx264 output_streamlit.mp4')
            os.system(
                'ffmpeg -y -i output_white.mp4 -vcodec libx264 output_white_streamlit.mp4')
        st.success('Done!')

        if options == 'Flow Field + Image':
            with open("output_streamlit.mp4", 'rb') as v:
                st.video(v)

        elif options == 'Only Flow Field':
            with open("output_white_streamlit.mp4", 'rb') as v:
                st.video(v)
        else:
            with open("output_streamlit.mp4", 'rb') as v:
                st.video(v)
            with open("output_white_streamlit.mp4", 'rb') as v:
                st.video(v)

   # Webcam Implementation

    while True:

        suc, img = cap.read()
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        prev_frame = frame

        if options == 'Flow Field + Image':
            FRAME_WINDOW2.image(viz_flow(frame, flow, step=steps,
                                         color=color_rgb, dot_radius=dot_radius)[1])

        elif options == 'Only Flow Field':
            FRAME_WINDOW1.image(viz_flow(frame, flow, step=steps,
                                         color=color_rgb, dot_radius=dot_radius)[0])
        else:

            FRAME_WINDOW1.image(viz_flow(frame, flow, step=steps,
                                         color=color_rgb, dot_radius=dot_radius)[0])

            FRAME_WINDOW2.image(viz_flow(frame, flow, step=steps,
                                         color=color_rgb, dot_radius=dot_radius)[1])


if __name__ == '__main__':
    main()
