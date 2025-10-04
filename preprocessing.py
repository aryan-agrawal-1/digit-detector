import numpy as np
import cv2
from scipy import ndimage
from PIL import Image

def preprocess_image(image):
    # Takes an image and converts it into the format our model expects
    
    # Convert PIL Image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # handle different lighting conditions
    # This makes the image black and white
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find the digit in the image by detecting contours
    digit_img = find_and_crop_digit(thresh)
    
    # If we couldn't find a digit, just use the whole thresholded image
    if digit_img is None:
        digit_img = thresh
    
    # Now we need to make it look like MNIST data
    # MNIST digits are centered and have padding
    processed = resize_and_pad(digit_img, size=20)  # Resize digit to 20x20
    processed = center_image(processed, canvas_size=28)  # Put it in a 28x28 canvas
    
    # Normalize pixel values to be between 0 and 1
    processed = processed.astype(np.float32) / 255.0
    
    # Flatten it into a column vector (784, 1) like our training data
    flattened = processed.reshape(784, 1)
    
    return flattened, processed  # Return both flattened for model and 28x28 for display


def find_and_crop_digit(binary_image):    
    # Find all contours (blobs) in the image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Find the largest contour (assuming that's our digit)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding box around this contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Make sure the bounding box is big enough to be a digit
    if w < 10 or h < 10:
        return None
    
    # Crop the image to just the digit with a small margin
    margin = 10
    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(binary_image.shape[1], x + w + margin)
    y_end = min(binary_image.shape[0], y + h + margin)
    
    cropped = binary_image[y_start:y_end, x_start:x_end]
    
    return cropped


def resize_and_pad(image, size=20):
    h, w = image.shape
    
    # Figure out which dimension is bigger so we can scale properly
    if h > w:
        new_h = size
        new_w = int(w * (size / h))
    else:
        new_w = size
        new_h = int(h * (size / w))
    
    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a square image with padding
    # Calculate padding needed on each side
    pad_h = (size - new_h) // 2
    pad_w = (size - new_w) // 2
    
    # Add padding to make it square
    padded = np.zeros((size, size), dtype=np.uint8)
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    return padded


def center_image(image, canvas_size=28):
    
    # Create the final 28x28 canvas
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    
    # Calculate center of mass of the digit
    cy, cx = ndimage.center_of_mass(image)
    
    # Calculate how much we need to shift to center it in the 28x28 canvas
    rows, cols = image.shape
    shiftx = canvas_size / 2 - cx
    shifty = canvas_size / 2 - cy
    
    # Create transformation matrix for shifting
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    
    # Apply the shift
    centered = cv2.warpAffine(image, M, (canvas_size, canvas_size))
    
    return centered


def preprocess_for_display(image_28x28):
    # Just scales it up so it's not tiny
    
    # Scale up to 280x280 so it's easier to see
    display_img = cv2.resize(image_28x28, (280, 280), interpolation=cv2.INTER_NEAREST)
    
    # Convert to uint8 for display
    display_img = (display_img * 255).astype(np.uint8)
    
    return display_img

