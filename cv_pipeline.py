import cv2
import matplotlib.pyplot as plt

def crop_image(image, width_percent, height_percent):
    if width_percent < 0 or width_percent > 1 or height_percent < 0 or height_percent > 1:
        raise ValueError("Percentage values must be between 0 and 1.")
    
    # Get the dimensions of the image
    height, width = image.shape
    
    # Calculate the new dimensions based on the specified percentages
    new_width = int(width * width_percent)
    new_height = int(height * height_percent)
    
    # Calculate the top-left coordinate of the cropped region
    x = int((width - new_width) / 2)
    y = int((height - new_height) / 2)
    
    # Crop the image
    cropped_image = image[y:y+new_height, x:x+new_width]
    
    return cropped_image

# Read the image
image = cv2.imread('test_image.jpeg')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply median filtering to reduce noise
median_image = cv2.medianBlur(gray_image, 11)

# Create the binary image
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Use an ellipse as the structuring element for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

# Close to remove noise (e.g., grid lines of the paper)
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=3)

# Erode to widen and join the path of the drawn number
eroded_image = cv2.erode(closed_image, kernel, iterations=16)
eroded_image = crop_image(eroded_image, .7, .7)

eroded_image = cv2.bitwise_not(eroded_image)

# Find contours
contours, _ = cv2.findContours(eroded_image, cv2.RETR_TREE, cv2.RETR_LIST)

# Create a copy of the eroded image for drawing contours
contour_image = cv2.cvtColor(eroded_image, cv2.COLOR_GRAY2BGR)

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Get the bounding box of the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)

# Calculate the maximum padding possible within the image boundaries
padding = 1000
top = max(0, y - padding)
bottom = min(eroded_image.shape[0], y + h + padding)
left = max(0, x - padding)
right = min(eroded_image.shape[1], x + w + padding)

# Crop the image to the bounding box with padding
cropped_image = eroded_image[y:y+h, x:x+w]

# Resize the cropped image to 32x32 pixels
resized_image = cv2.resize(cropped_image, (32, 32), interpolation=cv2.INTER_AREA)

# Display the final result
plt.imshow(resized_image, cmap='gray')
plt.title('Final image')
plt.show()
