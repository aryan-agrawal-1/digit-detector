import numpy as np
import cv2
from matplotlib import pyplot as plt
from preprocessing import preprocess_image, preprocess_for_display
from script import load_model, forward_prop, get_predictions

# This script tests if our preprocessing works correctly
# You can run this after training to see if it can predict a test image

def test_on_sample_image(image_path):    
    # Load the trained model weights
    print("Loading model...")
    W1, b1, W2, b2 = load_model('model_weights.npz')
    
    # Read the image
    print(f"Reading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Could not load image at {image_path}")
        return
    
    # Preprocess it
    print("Preprocessing image...")
    processed_flat, processed_28x28 = preprocess_image(image)
    
    # Make prediction
    print("Making prediction...")
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, processed_flat)
    prediction = get_predictions(A2)[0]
    
    # Get confidence scores for all digits
    probabilities = A2.flatten() * 100  # Convert to percentages
    
    # Display results
    print(f"Prediction: {prediction}")
    print("Confidence scores:")
    for digit in range(10):
        bar = '█' * int(probabilities[digit] / 2)  # Visual bar
        print(f"  {digit}: {probabilities[digit]:5.2f}% {bar}")
    
    # Show the original and preprocessed images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(processed_28x28, cmap='gray')
    ax2.set_title(f'Preprocessed (Prediction: {prediction})')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # You can test with any image file here
    # For now, let's create a simple test by taking a digit from the training data
    # and seeing if preprocessing + prediction works
    
    print("To test with your own image:")
    print("  test_on_sample_image('path/to/your/image.jpg')")
    print("\nMake sure you've trained the model first (run script.py)")

