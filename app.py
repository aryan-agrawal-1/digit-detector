import gradio as gr
import numpy as np
import cv2
from preprocessing import preprocess_image, preprocess_for_display
from script import load_model, forward_prop, get_predictions

# Load the model once when the app starts so we don't reload it every time
print("Loading trained model...")
try:
    W1, b1, W2, b2 = load_model('model_weights.npz')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Please run script.py first to train the model")
    exit(1)


def predict_digit(image):
    # This function gets called every time someone uploads an image or uses the webcam
    
    print(f"predict_digit called, image is None: {image is None}")
    
    if image is None:
        return "Please provide an image", None, None
    
    print(f"Image shape: {image.shape}")
    
    try:
        # Preprocess the image to make it look like MNIST data
        processed_flat, processed_28x28 = preprocess_image(image)
        print(f"Preprocessed shape: {processed_flat.shape}")
        
        # Run it through the neural network
        _, _, _, A2 = forward_prop(W1, b1, W2, b2, processed_flat)
        prediction = get_predictions(A2)[0]
        print(f"Prediction: {prediction}")
        
        # Get confidence scores for all digits
        probabilities = A2.flatten()
        
        # Create a dictionary for Label component
        confidence_dict = {str(i): float(probabilities[i]) for i in range(10)}
        
        # Scale up the preprocessed image so it's easier to see
        display_img = preprocess_for_display(processed_28x28)
        
        # Create a nice text output
        result_text = f"# Prediction: {prediction}\n\n"
        result_text += "### Confidence Scores:\n"
        
        # Sort by confidence and show top 3
        sorted_probs = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
        for i, (digit, prob) in enumerate(sorted_probs[:3]):
            result_text += f"{i+1}. **Digit {digit}**: {prob*100:.1f}%\n"
        
        return result_text, confidence_dict, display_img
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return f"Error processing image: {str(e)}", None, None


# Create the Gradio interface
with gr.Blocks(title="MNIST Digit Recognition") as demo:
    
    gr.Markdown("""
    # MNIST Digit Recognition
    
    Upload an image or use your webcam to recognise handwritten digits (0-9).
    """)
    
    with gr.Row():
        with gr.Column():
            # Input image from webcam or upload
            image_input = gr.Image(
                sources=["webcam", "upload"], 
                type="numpy",
                label="Capture a digit (0-9)",
                mirror_webcam=False
            )
            
            # Button to make prediction
            predict_btn = gr.Button("Recognise Digit", variant="primary", size="lg")
        
        with gr.Column():
            # prediction result
            result_text = gr.Markdown(label="Result")
            
            # confidence scores as a bar chart
            confidence_output = gr.Label(
                label="Confidence Distribution",
                num_top_classes=10
            )
    
    # Show the preprocessed image so user can see what the model actually sees
    with gr.Row():
        preprocessed_output = gr.Image(
            label="What the model sees (28x28 preprocessed)",
            type="numpy"
        )
    
    gr.Markdown("""
    ---
    ### About
    This is a 2-layer neural network trained from scratch on the MNIST dataset.
    The model preprocesses your image to match MNIST format.
    """)
    
    # Connect the button to the prediction function
    predict_btn.click(
        fn=predict_digit,
        inputs=image_input,
        outputs=[result_text, confidence_output, preprocessed_output]
    )
    
    # Also allow clicking the image itself to predict
    image_input.change(
        fn=predict_digit,
        inputs=image_input,
        outputs=[result_text, confidence_output, preprocessed_output]
    )


# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=True,  # Set to True if you want a public link
        server_name="127.0.0.1",
        server_port=7860
    )

