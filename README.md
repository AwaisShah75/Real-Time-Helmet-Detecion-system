# Real-Time Helmet Detection System

A real-time object detection system built to identify whether motorcyclists are wearing helmets using a custom-trained **YOLOv8s** model. The project features local real-time video inference, webcam support, and a Gradio web interface for testing.

## 🚀 Features

- **Custom YOLOv8 Model:** Trained on a specialized dataset to detect two specific classes: `With Helmet` and `Without Helmet`.
- **High-Performance Inference:** Utilizes `ultralytics` YOLOv8, `OpenCV`, and `cvzone` for smooth, real-time bounding box rendering and confidence score displays.
- **Flexible Video Sources:** Supports live inference via webcams or pre-recorded local video files (e.g., `traffic.mp4`).
- **Gradio Web Interface:** Includes a user-friendly Gradio web app for simple and quick inference testing.
- **Robust Training Pipeline:** Model was trained on Google Colab with advanced hyperparameters to prevent overfitting, using Google Drive for dataset management and persistence.

## 📁 Project Structure

- `live_inference.py`: The main script for running real-time inference on a webcam feed or video file.
- `Weights/best.pt`: The custom trained YOLOv8 model weights (Needs to be downloaded and placed here).
- `Helmet_Detection_YOLOv8s.ipynb`: The Jupyter Notebook containing the training pipeline used on Google Colab.
- `.gitignore`: Configured to keep the repository clean by excluding large dataset zip files and redundant folders.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd "Automatic numer plate"
   ```

2. **Install the required dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install ultralytics opencv-python cvzone math
   ```

3. **Download Model Weights:**
   Ensure your trained YOLOv8 weights file (`best.pt`) is placed inside a folder named `Weights/` in the root directory.

## 💻 Usage

To run the live inference script:

```bash
python live_inference.py
```

- **Video Source:** By default, the script reads from a local video (`Media/traffic.mp4`). To use your webcam, open `live_inference.py` and uncomment the `cap = cv2.VideoCapture(0)` line.
- **Exit:** Press **`q`** on your keyboard to close the live inference window.

## 🧠 Training Details

The model was trained using the YOLOv8s (small) architecture on Google Colab. Key training highlights include:
- Integrated manual dataset management with Google Drive.
- Implemented specific hyperparameters to aggressively prevent overfitting.
- Evaluated and tested via a custom Gradio web interface before local deployment.

## 📊 Empirical Multi-Model CPU Benchmark

As an extension of research following the NIJEC paper, this project includes an empirical benchmarking suite to evaluate the efficiency of the trained YOLOv8 model across different inference engines on a local CPU environment (Intel Xeon). 

The model was exported and evaluated using a custom benchmarking script (`colab_benchmark.py`) to measure the real-world trade-offs between speed, memory footprint, and model size.

### Benchmark Results (Intel Xeon CPU)

| Framework | Format | Size (MB) | Latency (ms) | FPS | RAM (MB) | mAP50 (%) |
|:---|:---|---:|---:|---:|---:|:---|
| PyTorch | `.pt` | 21.48 | 376.36 | 2.66 | — | 80.99 |
| ONNX Runtime | `.onnx` | 42.76 | 579.36 | 1.73 | 48.45 | 80.99 |
| TensorFlow Lite | `.tflite` | 42.69 | 701.82 | 1.42 | 56.88 | 80.99 |

> **Research Note:** Initial benchmarks incorrectly showed PyTorch as fastest due to automatic GPU selection in Google Colab. All formats were re-benchmarked with forced CPU inference to ensure fair comparison simulating resource-constrained deployment conditions. Benchmarked on Google Colab CPU (2-core). 300 frames, 10 warmup frames excluded from timing. ONNX and TFLite FP32 are lossless exports — mAP50 identical to PyTorch baseline. PyTorch RAM measurement unreliable in Colab environment due to shared process memory; model size used as proxy for deployment footprint.
