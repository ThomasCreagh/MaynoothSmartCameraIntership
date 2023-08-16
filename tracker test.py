import cv2
import numpy as np

# Define the dimensions of the black image in millimeters
area_width_mm = 15000
area_height_mm = 15000

# Define the resolution (pixels per millimeter)
resolution = 0.05  # You can adjust this based on your desired resolution

# Calculate the dimensions of the image in pixels
width = int(area_width_mm * resolution)
height = int(area_height_mm * resolution)

# Create a black image using NumPy
black_image = np.zeros((height, width, 3), dtype=np.uint8)

# Create a window for the image
cv2.namedWindow('Moving Circle')

# Initial position of the circle (in pixels)
circle_x = width // 2
circle_y = height // 2
circle_radius = 5

while True:
    # Copy the black image to work with a fresh canvas
    canvas = black_image.copy()
    
    # Draw a circle at the specified position
    cv2.circle(canvas, (circle_x, circle_y), circle_radius, (0, 0, 255), -1)
    
    # Display the image
    cv2.imshow('Moving Circle', canvas)
    
    # Wait for a key press; if 'q' is pressed, exit the loop
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    
    # Update the circle position (for example, move it to the right)
    circle_x = (circle_x + 5) % width

# Release the OpenCV window
cv2.destroyAllWindows()
