from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import random
#qoordinats for alarm box (x, y) in counterclockwise order
closedshape = [(1500, 3000), (2500, 2000), (2000, 1000), (1000, 1000), (5000, 2000)]


# Image dimensions in millimeters
image_width_mm = 4000
image_height_mm = 8000

image_width_px = 400
image_height_px = 800

# Draw grid lines
grid_spacing_mm = 1000
grid_color = (128, 128, 128)

# Define the resolution (pixels per millimeter)
resolution_x = image_width_px / image_width_mm 
resolution_y = image_height_px / image_height_mm

  
#pollygon testing
def ray_casting(point, shape):
    num_vertices = len(shape)
    inside = False

    for i in range(num_vertices):
        p1 = shape[i]
        p2 = shape[(i + 1) % num_vertices]

        if (p1[1] > point[1]) != (p2[1] > point[1]) and \
           point[0] < (p2[0] - p1[0]) * (point[1] - p1[1]) / (p2[1] - p1[1]) + p1[0]:
            inside = not inside

    return inside
# Convert millimeter coordinates to pixel coordinates
def mm_to_px_x(mm_coord):
    return int(mm_coord * resolution_x)

def mm_to_px_y(mm_coord):
    return int(mm_coord * resolution_y)

# Dictionary to store circle information
circles = {}

# Function to generate a random BGR color
def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Function to add/update circle in the dictionary
def update_circle(circle_id, x_mm, y_mm, status):
    x_px = mm_to_px_x(x_mm + image_width_mm/2)
    y_px = mm_to_px_y(image_height_mm - y_mm)
    circle_radius_px = 10
    if status == "REMOVED":
        if circle_id in circles:
            del circles[circle_id]

    else:
        circle_color = generate_random_color()
        if circle_id in circles:
            circle_color = circles[circle_id][2]

        if status == "LOST":
            circle_color = (0, 0, 255)  # Black for lost circles
            circle_radius_px = 10  # Adjust the radius as needed
           
        circles[circle_id] = (x_px, y_px,circle_color, circle_radius_px)

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

nnPathDefault = str((Path(__file__).parent / Path('../models/mobilenet-ssd_openvino_2021.4_5shave.blob')).resolve().absolute())
parser = argparse.ArgumentParser()
parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
parser.add_argument('-ff', '--full_frame', action="store_true", help="Perform tracking on full RGB frame", default=False)

args = parser.parse_args()

fullFrameTracking = args.full_frame

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
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
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
camRgb.setIspScale(1,3) # You don't need to downscale (4k -> 720P) video frames
# Crop video to match aspect ratio of aspect ratio of preview (1:1)
camRgb.setVideoSize(720,720)

xoutFrames = pipeline.create(dai.node.XLinkOut)
xoutFrames.setStreamName("frames")
camRgb.video.link(xoutFrames.input)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

# setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Align depth map to the perspective of RGB camera, on which inference is done
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

spatialDetectionNetwork.setBlobPath(args.nnPath)
spatialDetectionNetwork.setConfidenceThreshold(0.75)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(15000)

objectTracker.setDetectionLabelsToTrack([5,15])  # track only people and bottle 
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
objectTracker.out.link(trackerOut.input)

if fullFrameTracking:
    camRgb.setPreviewKeepAspectRatio(False)
    camRgb.video.link(objectTracker.inputTrackerFrame)
    objectTracker.inputTrackerFrame.setBlocking(False)
    # do not block the pipeline if it's too slow on full frame
    objectTracker.inputTrackerFrame.setQueueSize(2)
else:
    spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
spatialDetectionNetwork.out.link(objectTracker.inputDetections)
stereo.depth.link(spatialDetectionNetwork.inputDepth)

# Connect to device and start pipeline
with dai.Device(pipeline, usb2Mode=True ) as device:
    qFrames = device.getOutputQueue(name="frames")
    preview = device.getOutputQueue("preview", 4, False)
    tracklets = device.getOutputQueue("tracklets", 4, False)

    startTime = time.monotonic()
    counter = 0
    fps_1 = 0
    color = (255, 255, 255)

    while(True):
        imgFrame = preview.get()
        track = tracklets.get()
        frameLarge = qFrames.get().getCvFrame()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps_1 = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        ## STEP 1 CREATE BEV IMAGE

        # Create a blank black image
        image = np.zeros((image_height_px, image_width_px, 3), dtype=np.uint8)
        for x in range(0, image_width_mm, grid_spacing_mm):
                x_px = mm_to_px_x(x)
                cv2.line(image, (x_px, 0), (x_px, image.shape[0]), grid_color, 1)

        for y in range(0, image_height_mm, grid_spacing_mm):
            y_px = mm_to_px_y(y)
            cv2.line(image, (0, y_px), (image.shape[1], y_px), grid_color, 1)
        polygon_points = np.array(closedshape, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [polygon_points], True, (0, 12, 255))

        frame = imgFrame.getCvFrame()
        trackletsData = track.tracklets
        for t in trackletsData:
            roi = t.roi.denormalize(frameLarge.shape[1], frameLarge.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            try:
                label = labelMap[t.label]
            except:
                label = t.label

            cv2.putText(frameLarge, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frameLarge, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frameLarge, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frameLarge, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

            cv2.putText(frameLarge, f"X: {int(t.spatialCoordinates.x)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frameLarge, f"Y: {int(t.spatialCoordinates.y)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frameLarge, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            #if (int(t.spatialCoordinates.x) >= 70):
            #    if label == "bottle":
            #        print("warning tresspasser with bottle")
            #    else:
            #        print("trespasser")
            print("people id is", int(t.id), "x", int(t.spatialCoordinates.x) + image_width_mm/2, "z", int(t.spatialCoordinates.z), "name" , t.status.name,)
            boxtestpoint = (int(t.spatialCoordinates.x)+ image_width_mm/2, int(t.spatialCoordinates.z))

            if ray_casting(boxtestpoint, closedshape):
                print("person is inside the shape.")
            ## STEP 2 Draw circle for current ID on blank image

            # Draw the moving point
            peopleID = int(t.id)
            update_circle(peopleID, int(t.spatialCoordinates.x), int(t.spatialCoordinates.z), t.status.name)


        cv2.putText(frameLarge, "NN fps_1: {:.2f}".format(fps_1), (2, frameLarge.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

        cv2.imshow("tracker", frameLarge)
        # Draw circles on the image
        for circle_id, (x_px, y_px, color, radius) in circles.items():
            cv2.circle(image, (x_px, y_px), radius, color, -1)
            #we can have the lines appear with the circle
            # for x in range(0, image_width_mm, grid_spacing_mm):
            #     x_px = mm_to_px_x(x)
            #     cv2.line(image, (x_px, 0), (x_px, image.shape[0]), grid_color, 1)

            # for y in range(0, image_height_mm, grid_spacing_mm):
            #     y_px = mm_to_px_y(y)
            #     cv2.line(image, (0, y_px), (image.shape[1], y_px), grid_color, 1)

        # Display the image with the moving point
        cv2.imshow("Circles Image", image)

        ## STEP 3 IMSHOW of BEV Image
        if cv2.waitKey(1) == ord('q'):
            break