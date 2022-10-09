import numpy as np
import cv2


def viz_flow(img, flow, step=32, color=(0, 255, 0), dot_radius=1):
    height, width = img.shape[:2]  # Get the height and width of image frame
    y, x = np.mgrid[step/2:height:step, step /
                    2:width:step].reshape(2, -1).astype(int)  # Define a 2D meshgrid
    fx, fy = flow[y, x].T  # Extracting x and y component of the flow-field
    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    # Conversion from Gray to BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    whiteFrame = 255 * \
        np.ones((img_bgr.shape[0], img_bgr.shape[1], 3), np.uint8)
    cv2.polylines(whiteFrame, lines, 10, color)
    cv2.polylines(img_bgr, lines, 10, color)

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(whiteFrame, (x1, y1), dot_radius, color, -1)
        cv2.circle(img_bgr, (x1, y1), dot_radius, color, -1)

    return whiteFrame, img_bgr
