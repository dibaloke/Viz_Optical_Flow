# Visualize Optical Flow 

This project makes use of [OpenCV](https://opencv.org/) and [Streamlit](https://streamlit.io/) to Visualize Optical Flow from Webcam Input and Uploaded Video.

## How to Run 

- Clone the repository
- Make sure you have all the dependancies installed from requirments.text
- In addition, make sure you have a stable version of ffmpeg installed as ffmpeg is utilized to generate browser compatiable 
- Run `` streamlit run app.py `` 
## Optical Flow From Webcam Input

Takes real time input from webcam and generates stream of Optical Flow. 

![](read_me_images/Webcam_Implementation.png)

- Select "Flow Field+Image" to overlay optical flow on top of the video input
- Select "Only Flow Field" to see only the flow field
- Select "Both" to see them both

[Checkout how it works !](https://youtu.be/sk2q45UMneg)

## Optical Flow From Uploaded Video
![](read_me_images/Video_Implementation1.png)


Upload a video by selecting a file or by dragging and dropping.

![](read_me_images/Video_Implementation2.png)

Meta Data like Number of Frames, Video Duration, FPS will be shown.


![](read_me_images/Video_Implementation3.png)

[Check Out how it works](https://youtu.be/k4KqhUSRSuY)