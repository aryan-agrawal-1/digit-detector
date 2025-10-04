# MNIST Digit Recognition from Scratch

A simple 2-layer neural network trained from scratch using only numpy to recognise handwritten digits.

I started by following this YouTube tutorial:
https://www.youtube.com/watch?v=w8yWXqWQYmU

And then added a frontend so I could see how well the image detection worked on my writing.

Datasets are not included (too large for git) - you can find them from the YT video above (they can be found on kaggle)

## Features
- Neural network implemented with just NumPy
- Web interface with camera/upload support
- Real-time digit recognition

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. **Train the model** (takes a few minutes):
```bash
python script.py
```

2. **Launch the web app**:
```bash
python app.py
```

3. Open `http://127.0.0.1:7860` in your browser

## Tips for Best Results
- Write one digit on white paper with black pen/marker
- Fill most of the frame with the digit
- Ensure good lighting, clean background

## Architecture
- Input: 784 (28×28 grayscale image)
- Hidden layer: 10 neurons (ReLU)
- Output: 10 neurons (Softmax)
- Training: 59,000 images, 500 iterations
- Validation accuracy: ~85-90%

