# Smart Vehicle Counter with YOLO & SORT Tracking

An advanced real-time vehicle counting system that uses YOLOv8 for object detection and SORT algorithm for multi-object tracking. The system can accurately count different types of vehicles crossing predefined counting lines in video streams.

## 🎬 Demo

### 🔄 Before & After Comparison

#### 🎥 Original Video (cars.mp4)
![Original](cars.gif)

#### 🧠 After Processing (detection_demo.mp4)
![Processed](detection_demo.gif)




### What You'll See in the Demo:
- 🎯 **Real-time Detection**: Bounding boxes around detected vehicles
- 🏷️ **Vehicle Classification**: Labels showing car, truck, bus, motorbike
- 🔢 **Unique ID Tracking**: Each vehicle gets a persistent tracking ID
- 📈 **Live Statistics**: Real-time count display by vehicle type
- 🚦 **Counting Line**: Red line that triggers counting when crossed
- 🌟 **Vehicle Trajectories**: Colored trails showing movement paths
- ⚡ **Performance Metrics**: FPS counter in top-right corner

## 🚗 Features

- **Real-time Vehicle Detection**: Uses YOLOv8 for accurate vehicle detection
- **Multi-Object Tracking**: Implements SORT algorithm for consistent vehicle tracking
- **Vehicle Classification**: Distinguishes between cars, trucks, buses, and motorbikes
- **Adaptive Confidence Thresholds**: Different confidence levels for different vehicle types
- **Visual Trajectory Tracking**: Shows vehicle paths with fading trail effects
- **Customizable Counting Lines**: Support for multiple counting lines with direction detection
- **Region of Interest (ROI)**: Optional mask support for focusing on specific areas
- **Real-time Statistics**: Live display of counts by vehicle type and FPS
- **Screenshot Capability**: Save detection results with statistics

## 📋 Requirements

```bash
pip install ultralytics
pip install opencv-python
pip install cvzone
pip install torch
pip install numpy
```

### Additional Dependencies
- **SORT Tracker**: Download `sort.py` from [SORT GitHub Repository](https://github.com/abewley/sort)

## 🗂️ Project Structure

```
smart-vehicle-counter/
├── main.py                 # Main vehicle counter script
├── mask.png                # ROI mask (optional)
├── cars.mp4                # Original input video file
├── detection_demo.mp4      # Processed video with detection overlay
├── requirements.txt        # Python dependencies
└── README.md              # This comprehensive guide
```

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/NoobML/smart-vehicle-counter.git
   cd smart-vehicle-counter
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO weights**
   - The script will automatically download YOLOv8 weights on first run
   - Or manually download from [Ultralytics](https://github.com/ultralytics/ultralytics)

4. **Download SORT tracker**
   ```bash
   wget https://raw.githubusercontent.com/abewley/sort/master/sort.py
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

## ⚙️ Configuration

### Vehicle Classes and Confidence Thresholds

The system uses different confidence thresholds for optimal detection:

```python
vehicle_classes = {
    'car': 0.3,        # Standard confidence for cars
    'truck': 0.25,     # Lower threshold for trucks (larger, easier to detect)
    'bus': 0.25,       # Lower threshold for buses
    'motorbike': 0.35, # Higher threshold for motorbikes (smaller, harder to detect)
}
```

### Counting Lines Configuration

Modify counting lines in the `__init__` method:

```python
self.counting_lines = [
    {'coords': [x1, y1, x2, y2], 'name': 'Line1', 'direction': 'horizontal'},
    {'coords': [x1, y1, x2, y2], 'name': 'Line2', 'direction': 'vertical'}
]
```

### SORT Tracker Parameters

```python
self.tracker = Sort(
    max_age=50,        # Maximum frames to keep track without detection
    min_hits=2,        # Minimum detections before confirming track
    iou_threshold=0.2  # IoU threshold for matching detections
)
```

## 🔧 Class Documentation

### `VehicleCounter`

Main class that handles vehicle detection, tracking, and counting.

#### `__init__(self, video_path, model_path, mask_path=None)`
Initializes the vehicle counter with video source and model.

**Parameters:**
- `video_path` (str): Path to input video file
- `model_path` (str): Path to YOLO model weights
- `mask_path` (str, optional): Path to ROI mask image

#### `is_valid_vehicle(self, class_name, confidence)`
Validates if a detection is a vehicle with sufficient confidence.

**Parameters:**
- `class_name` (str): Detected object class name
- `confidence` (float): Detection confidence score

**Returns:**
- `bool`: True if valid vehicle detection

#### `check_line_crossing(self, center_x, center_y, vehicle_id, line_info)`
Detects when a vehicle crosses a counting line.

**Parameters:**
- `center_x, center_y` (int): Vehicle center coordinates
- `vehicle_id` (int): Unique tracking ID
- `line_info` (dict): Line configuration with coordinates and direction

**Returns:**
- `bool`: True if line crossing detected

#### `draw_statistics(self, img)`
Renders real-time statistics on the video frame.

**Parameters:**
- `img` (numpy.ndarray): Video frame to draw on

#### `process_frame(self)`
Processes a single video frame for detection and tracking.

**Returns:**
- `tuple`: (processed_frame, success_flag)

#### `run(self)`
Main execution loop that processes the entire video stream.

## 🎮 Controls

- **'q'**: Quit the application
- **'s'**: Save screenshot with current statistics

## 📊 Output

The system provides:
- **Real-time counting**: Live vehicle count display
- **Vehicle classification**: Separate counts for different vehicle types
- **Performance metrics**: FPS display
- **Visual feedback**: 
  - Green rectangles for tracked vehicles
  - Purple rectangles for new detections
  - Vehicle trajectories with fading trails
  - Line crossing animations

## 🎯 Use Cases

- **Traffic Analysis**: Monitor traffic flow at intersections
- **Road Planning**: Collect data for infrastructure decisions
- **Security Systems**: Vehicle monitoring for restricted areas
- **Research**: Traffic pattern analysis and studies
- **Smart Cities**: Integration with traffic management systems

## 🔬 Technical Details

### Detection Pipeline
1. **Frame Preprocessing**: Apply ROI mask if provided
2. **YOLO Detection**: Run YOLOv8 inference on frame
3. **Filtering**: Apply confidence thresholds per vehicle type
4. **Tracking**: Update SORT tracker with detections
5. **Line Crossing**: Check vehicle positions against counting lines
6. **Visualization**: Render bounding boxes, trajectories, and statistics

### Performance Optimizations
- **Confidence Thresholding**: Reduces false positives
- **Trajectory Limiting**: Keeps only recent 30 positions per vehicle
- **Efficient Rendering**: Optimized drawing operations
- **Memory Management**: Automatic cleanup of old tracking data

## 🛡️ Troubleshooting

### Common Issues

1. **Low FPS Performance**
   - Use smaller YOLO model (yolov8n.pt instead of yolov8l.pt)
   - Reduce video resolution
   - Adjust confidence thresholds

2. **Missed Vehicles**
   - Lower confidence thresholds
   - Adjust SORT parameters (min_hits, max_age)
   - Check counting line positioning

3. **False Positives**
   - Increase confidence thresholds
   - Use ROI mask to focus on road areas
   - Fine-tune SORT IoU threshold

4. **Memory Issues**
   - Limit trajectory history length
   - Process video in batches
   - Use GPU acceleration if available

## 📈 Performance Tips

- **GPU Acceleration**: Ensure PyTorch with CUDA for faster inference
- **Video Resolution**: Balance between accuracy and speed
- **Model Selection**: Choose appropriate YOLO model size
- **ROI Usage**: Use masks to focus on relevant areas

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -am 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Alex Bewley](https://github.com/abewley/sort) for SORT tracking algorithm
- [CVZone](https://github.com/cvzone/cvzone) for computer vision utilities

## 📞 Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section
2. Open an issue on GitHub
3. Provide video samples and error logs for faster resolution

---

**Made with ❤️ for the Computer Vision Community**
