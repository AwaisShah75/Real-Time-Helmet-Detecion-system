import time
import os
import pandas as pd
from ultralytics import YOLO
import cv2
import gc
import torch
import psutil

# ---------------------------------------------------------
# YOLOv8 Multi-Model CPU Benchmark Script
# ---------------------------------------------------------

# --- USER CONFIG ---
# Change these paths to run on any machine or Colab
VIDEO_PATH = "/content/drive/MyDrive/Helmet_Detection_Project/bike_1.mp4" 
WEIGHTS_DIR = "/content/drive/MyDrive/Helmet_Detection_Project/runs/yolov8s_helmet/weights"
FRAMES_TO_TEST = 300  # Number of frames to test (300 frames @ 30fps = ~10 seconds)

# --- 1. SETUP ---
print("--- Hardware Check ---")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("----------------------\n")

print(f"Loading up to {FRAMES_TO_TEST} video frames into RAM...")
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
while len(frames) < FRAMES_TO_TEST:
    ret, frame = cap.read()
    if not ret: 
        break
    frames.append(frame)
cap.release()
print(f"Successfully loaded {len(frames)} frames for testing.")

# --- 2. BENCHMARKING LOGIC ---
def benchmark_model(model_path, model_name, format_name):
    print(f"\n--- Benchmarking {model_name} ({format_name}) ---")
    
    # Force Python Garbage Collection to clear memory from previous models
    gc.collect()
    
    # Calculate Model Size
    model_size_mb = round(os.path.getsize(model_path) / (1024 * 1024), 2)
    
    # Initialize psutil
    process = psutil.Process(os.getpid())

    # Before load
    ram_before = process.memory_info().rss / (1024 * 1024)

    # Load model (explicitly defining task='detect' to fix the ONNX warning)
    model = YOLO(model_path, task='detect')
    model.overrides['device'] = 'cpu'

    # Warm it once so native backend fully initializes memory buffers
    _ = model.predict(frames[0], verbose=False, device='cpu')
    time.sleep(0.5)  # Let allocations settle

    # After load
    ram_after = process.memory_info().rss / (1024 * 1024)
    ram_footprint = round(ram_after - ram_before, 2)
    
    # 3. Warm-up (AI models have 'cold start' times on the first few frames)
    # We already did 1 frame for memory, so let's do 9 more.
    print("Warming up engine (remaining frames)...")
    for i in range(9):
        _ = model.predict(frames[0], verbose=False, device='cpu')
        
    # 4. Actual Benchmark (The Stopwatch Starts)
    print(f"Running inference on {len(frames)} frames...")
    start_time = time.perf_counter()
    
    for frame in frames:
        _ = model.predict(frame, verbose=False, device='cpu')
        
    end_time = time.perf_counter()
    # (Stopwatch Stops)
    
    # Clean up model from memory
    del model
    gc.collect()
    
    # 5. Calculations
    total_time = end_time - start_time
    avg_latency_ms = (total_time / len(frames)) * 1000
    fps = len(frames) / total_time
    
    print(f"Result -> Latency: {avg_latency_ms:.2f} ms | FPS: {fps:.2f} | Peak RAM Load: {ram_footprint:.2f} MB")
    
    return {
        "Framework": model_name,
        "Format": format_name,
        "Size (MB)": model_size_mb,
        "Average Latency (ms)": round(avg_latency_ms, 2),
        "FPS": round(fps, 2),
        "RAM Footprint (MB)": ram_footprint,
        "mAP50 (%)": "TBD"  # Fill manually
    }

# --- 3. RUN THE TESTS ---
results = []

try:
    # Test PyTorch (baseline)
    results.append(benchmark_model(f'{WEIGHTS_DIR}/best.pt', 'PyTorch', '.pt'))
    
    # Test ONNX
    results.append(benchmark_model(f'{WEIGHTS_DIR}/best.onnx', 'ONNX Runtime', '.onnx'))
    
    # Test TFLite (Ultralytics automatically exports to a sub-folder)
    results.append(benchmark_model(f'{WEIGHTS_DIR}/best_saved_model/best_float32.tflite', 'TensorFlow Lite', '.tflite'))
    
except Exception as e:
    print(f"\nError running benchmark: {e}")
    print("Did you make sure to run the export cell first and upload your video?")

# --- 4. SHOW FINAL RESEARCH TABLE ---
if results:
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("                        FINAL BENCHMARK RESULTS")
    print("="*80)
    print(df.to_markdown(index=False))
    
    print("\nNOTE: Since ONNX and TFLite (FP32) are mathematically equivalent to the PyTorch")
    print("baseline, you only need to calculate mAP50 once on the .pt model. The other two")
    print("formats will share the exact same mAP50 value.")
    print("\nRun this cell after the benchmark to get your mAP50:")
    print("  model = YOLO('path_to_format/best.pt')")
    print("  metrics = model.val(data='/content/drive/MyDrive/Helmet_Detection_Project/data.yaml', verbose=False, device='cpu')")
    print("  print(f'mAP50: {metrics.box.map50 * 100:.2f}%')")

    # Save to CSV for easy import into Excel / Research Paper
    df.to_csv('benchmark_results.csv', index=False)
    print("\n✅ Saved to 'benchmark_results.csv'! You can download this file.")
