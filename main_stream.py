#!/usr/bin/env python3

import cv2
import depthai as dai
from flask import Flask, Response
from main_class import SmartCamera

app = Flask(__name__)
smart_cam = SmartCamera()

# # DepthAI pipeline
# pipeline = dai.Pipeline()

# camRgb = pipeline.create(dai.node.ColorCamera)
# xoutVideo = pipeline.create(dai.node.XLinkOut)

# xoutVideo.setStreamName("video")

# camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# camRgb.setVideoSize(1920, 1080)

# xoutVideo.input.setBlocking(False)
# xoutVideo.input.setQueueSize(1)

# camRgb.video.link(xoutVideo.input)

# # DepthAI device
# device = dai.Device(pipeline)

# def generate_frames():
#     video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
#     while True:
#         videoIn = video.get()
#         frame = videoIn.getCvFrame()
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return """<!DOCTYPE html>
    <html>
    <head>
        <title>DepthAI Video Stream</title>
    </head>
    <body>
        <h1>DepthAI Video Stream</h1>
        <img src="/video_feed" />
    </body>
    </html>"""

@app.route('/video_feed')
def video_feed():
    return Response(smart_cam.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
