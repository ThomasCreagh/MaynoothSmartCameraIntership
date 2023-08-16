#!/usr/bin/env python3

from pathlib import Path
import depthai as dai
import numpy as np
import time
import argparse
import cv2
import blobconverter
from utility import *
#label texts
print("label texts")
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

#parsing stuff
print("parsing stuff")
nnPathDefault = str((Path(__file__).parent / Path('../models/mobilenet-ssd_openvino_2021.4_5shave.blob')).resolve().absolute())
parser = argparse.ArgumentParser()
parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
parser.add_argument('-ff', '--full_frame', action="store_true", help="Perform tracking on full RGB frame", default=False)

args = parser.parse_args()

fullFrameTracking = args.full_frame

# Create pipeline ---
print("Creating pipeline")
pipeline = dai.Pipeline()
camRgb = pipeline.create(dai.node.ColorCamera)

# Define sources and outputs
print("Defining sources and outputs")
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
camRgb.setIspScale(1,3) # You don't need to downscale (4k -> 720P) video frames
# Crop video to match aspect ratio of aspect ratio of preview (1:1)
camRgb.setVideoSize(720,720)
#creates big frame
xoutFrames = pipeline.create(dai.node.XLinkOut)
xoutFrames.setStreamName("frames")
camRgb.video.link(xoutFrames.input)
#for distance
print("setting up distance")
spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
objectTracker = pipeline.create(dai.node.ObjectTracker)

xoutRgb = pipeline.create(dai.node.XLinkOut)
trackerOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("preview")
trackerOut.setStreamName("tracklets")




# Properties
print("setting up Properties")
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

# setting node configs
print("setting node configs")
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Align depth map to the perspective of RGB camera, on which inference is done
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

spatialDetectionNetwork.setBlobPath(args.nnPath)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

objectTracker.setDetectionLabelsToTrack([5,9,15])  # track only person
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

# Linking
print("linking up modules")
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
objectTracker.out.link(trackerOut.input)

if fullFrameTracking:
    camRgb.setPreviewKeepAspectRatio(False)
    #problem here with the .video notation
    #camRgb.video.link(objectTracker.inputTrackerFrame)
    objectTracker.inputTrackerFrame.setBlocking(False)
    # do not block the pipeline if it's too slow on full frame
    objectTracker.inputTrackerFrame.setQueueSize(2)
else:
    spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
spatialDetectionNetwork.out.link(objectTracker.inputDetections)
stereo.depth.link(spatialDetectionNetwork.inputDepth)

# Define a neural network that will make predictions based on the source frames
print("making predictions")
nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
camRgb.preview.link(nn.input)

passthroughOut = pipeline.create(dai.node.XLinkOut)
passthroughOut.setStreamName("pass")
nn.passthrough.link(passthroughOut.input)

nnOut = pipeline.create(dai.node.XLinkOut)
nnOut.setStreamName("nn")
nn.out.link(nnOut.input)

# Connect to device and start pipeline
print("connecting to device")
with dai.Device(pipeline, usb2Mode=True) as device:
    qFrames = device.getOutputQueue(name="frames")
    qPass = device.getOutputQueue(name="pass")
    qDet = device.getOutputQueue(name="nn")

    detections = []
    fps = FPSHandler()
    text = TextHelper() 

    preview = device.getOutputQueue("preview", 4, False)
    tracklets = device.getOutputQueue("tracklets", 4, False)

    startTime = time.monotonic()
    counter = 0
    fps_1 = 0
    color = (255, 255, 255)
    # nn data (bounding box locations) are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
    
    def displayFrame(name, frame):
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            text.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20))
            text.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40))
            text.rectangle(frame, bbox)
        # Show the frame
        cv2.imshow(name, frame)

    while(True):
        imgFrame = preview.get()
        track = tracklets.get()

        counter+=1
        current_time = time.monotonic()

        frame = (qFrames.get()).getCvFrame()

        inDet = qDet.tryGet()
        if inDet is not None:
            detections = inDet.detections
            fps.next_iter()
        inPass = qPass.tryGet()

        if (current_time - startTime) > 1 :
            fps_1 = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
        
        #this can turn on and off the passthrough box
        if inPass is not None:
            displayFrame('Passthrough', inPass.getCvFrame())

        # from here ↓
        frame1 = imgFrame.getCvFrame()
        trackletsData = track.tracklets
        for t in trackletsData:
            roi = t.roi.denormalize(frame1.shape[1], frame1.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            try:
                label = labelMap[t.label]
            except:
                label = t.label

            cv2.putText(frame1, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame1, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame1, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame1, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

            #here we can find the x y z quordinates
            cv2.putText(frame1, f"X: {int(t.spatialCoordinates.x)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame1, f"Y: {int(t.spatialCoordinates.y)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame1, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)


        #turning these off turns off just the tracker display 
        cv2.putText(frame1, "NN fps_1: {:.2f}".format(fps_1), (2, frame1.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

        cv2.imshow("tracker", frame1)
        #to here we can turn on and off the tracker module



        # Draw bounding boxes on high-res frame and show it
        text.putText(frame, "NN fps: {:.2f}".format(fps.fps()), (2, frame.shape[0] - 4))
        displayFrame("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            print("breaking away")
            break
