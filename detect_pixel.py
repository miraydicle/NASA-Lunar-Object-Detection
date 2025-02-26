import cv2
import numpy as np

# Load the image
image_path = "C:/Users/Dell/Desktop/NASA_LAC/all_images/semantic/15606semantic.png"
semantic_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resize for better visibility
height, width = semantic_img.shape
scale = 800 / max(height, width)  # Scale factor
semantic_img_resized = cv2.resize(semantic_img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_NEAREST)

# Convert image to color to display text
semantic_img_color = cv2.cvtColor(semantic_img_resized, cv2.COLOR_GRAY2BGR)

# Loop through the image and overlay pixel values
for y in range(0, semantic_img_resized.shape[0], 30):  # Step size 30 pixels
    for x in range(0, semantic_img_resized.shape[1], 30):
        pixel_value = semantic_img_resized[y, x]
        cv2.putText(semantic_img_color, str(pixel_value), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# Show the image
cv2.imshow("Semantic Image with Pixel Values", semantic_img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
