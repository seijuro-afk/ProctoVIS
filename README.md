# ProctoVIS - AI-Powered Exam Proctoring System

A computer vision-based exam proctoring system that monitors test-taker behavior in real-time to detect anomalies and potential academic integrity violations.

## Overview

ProctoVIS is an intelligent surveillance system designed for remote exam proctoring. It uses advanced computer vision and deep learning to:

- **Track facial features** and detect when test-takers are present and attentive
- **Monitor gaze direction** to detect screen-switching or looking away from the exam
- **Estimate head pose** to identify unusual behavior or excessive head movement
- **Detect environmental anomalies** including lighting changes, phone usage, and video skipping
- **Generate comprehensive reports** with timestamp-marked incidents for review

## Features

### Core Detection Capabilities
- **Face Detection & Tracking**: Robust facial detection using cascade classifiers and YOLO-v8
- **Facial Landmarks**: Precise landmark detection using ONNX-based neural networks
- **Gaze Estimation**: Deep learning-based gaze direction using L2CS-Net (ResNet50)
- **Head Pose Estimation**: 3D head orientation tracking (yaw, pitch, roll)
- **Pupil Centroid Localization**: Precise eye position tracking

### Anomaly Detection
- **Face Absence Detection**: Alerts when test-taker leaves the frame for extended periods
- **Gaze Deviation Detection**: Identifies when gaze deviates from the test screen
- **Head Pose Anomalies**: Detects unusual head positioning or rotation
- **Lighting Changes**: Monitors for sudden lighting variations (potential alt-tab indicators)
- **Phone Detection**: Identifies phone presence in the frame
- **Video Skip Detection**: Detects frozen frames or video discontinuities
- **Duplicate Face Detection**: Alerts when multiple faces appear in the frame

### Output
- **Real-time Visualization**: Visual overlay with landmarks, gaze vectors, and alerts
- **Video Recording**: Output video with annotations and alerts
- **Incident Logging**: Timestamped log of all detected anomalies with severity levels
- **Analysis Reports**: Comprehensive proctoring reports for examination review

## Project Structure

```
ProctoVIS/
├── main_proctor.m              # Main MATLAB entry point and processing loop
├── gaze_engine.py              # Python gaze estimation module (L2CS-Net wrapper)
├── landmarkFcn.m               # Facial landmark extraction
├── onnxLandmarkFcn.m           # ONNX model interface for landmarks
│
├── modules/                    # MATLAB detection modules
│   ├── detectAnomalies.m       # Main anomaly detection logic
│   ├── detectFaceLightChange.m # Lighting change detection
│   ├── detectPhoneLight.m      # Phone detection
│   ├── detectVideoSkip.m       # Video skip/freeze detection
│   ├── estimateGaze.m          # Gaze direction estimation
│   ├── estimateHeadPose.m      # 3D head pose calculation
│   ├── generateReport.m        # Report generation
│   ├── getFacialLandmarks.m    # Landmark processing
│   ├── getReferenceData.m      # Reference frame analysis
│   ├── initFaceLightState.m    # Initialize lighting state tracker
│   ├── locatePupilCentroid.m   # Pupil detection and tracking
│   └── poseSignalTools.m       # Pose signal processing utilities
│
├── models/                     # Pre-trained neural network models
│   ├── face_landmarks.onnx     # Facial landmark detection model
│   └── L2CSNet_gaze360.pkl     # L2CS-Net gaze estimation model
│
├── data/                       # Test data and sample videos
│   └── *.mp4                   # Sample video files for testing
│
└── resources/                  # MATLAB project configuration
```

## Technical Stack

| Component | Technology |
|-----------|-----------|
| **Main Framework** | MATLAB (Computer Vision Toolbox) |
| **Gaze Estimation** | Python + PyTorch + L2CS-Net |
| **Face Detection** | Cascade Classifiers + YOLO-v8 |
| **Facial Landmarks** | ONNX Neural Network |
| **Head Pose Estimation** | Perspective-n-Point (PnP) with camera intrinsics |
| **Video I/O** | MATLAB VideoReader/VideoWriter |
| **ML Framework** | PyTorch (for gaze pipeline) |

## Installation

### Prerequisites

- **MATLAB R2019b or later** with:
  - Computer Vision Toolbox
  - Deep Learning Toolbox
- **Python 3.8+** with:
  - PyTorch (CPU or GPU)
  - OpenCV
  - L2CS-Net library
- **Operating System**: Windows 10/11 (tested on Windows)

### Setup Steps

1. **Clone or download the repository**
   ```
   cd ProctoVIS
   ```

2. **Configure Python Environment** (if not using system Python)
   
   Edit `main_proctor.m` and uncomment the Python engine configuration:
   ```matlab
   pathToPython = 'D:\Anaconda\envs\proctor_env\python.exe'; 
   pyenv('Version', pathToPython, 'ExecutionMode', 'OutOfProcess');
   ```

3. **Install Python Dependencies**
   ```bash
   pip install torch torchvision opencv-python l2cs-net
   ```

4. **Prepare Test Video**
   
   Place your test video in the `data/` directory and update the path in `main_proctor.m`:
   ```matlab
   videoPath = 'data\your_video.mp4';
   ```

5. **Configure Output Path**
   
   Update the output video path in `main_proctor.m`:
   ```matlab
   vWriter = VideoWriter('proctor_output_subject1.mp4', 'MPEG-4');
   ```

## Usage

### Running the Proctoring Analysis

1. Open MATLAB and navigate to the ProctoVIS directory
2. Run the main script:
   ```matlab
   main_proctor
   ```

3. The system will:
   - Load video and initialize all models
   - Process each frame in real-time
   - Detect anomalies and log incidents
   - Write annotated video to output file
   - Display processing status and statistics

### Configuration Parameters

Key parameters in `main_proctor.m` that can be adjusted:

```matlab
procScale = 0.5;                    % Processing scale (0.5 = 50% resolution for speed)
detectEvery = 4;                    % Process every Nth frame
useLowResProcessing = true;         % Enable low-res mode for speed
reportIntervalSec = 2;              % Incident report interval (seconds)
faceAbsenceThresholdSec = 3;        % Face absence alarm threshold
gazeDeviationThresholdSec = 3;      % Gaze deviation alarm threshold
headPoseDeviationThresholdSec = 3;  % Head pose alarm threshold
```

### Screen Region Configuration

Define the test screen region for gaze tracking (in pixels):
```matlab
screenLeft = 300; 
screenRight = 900; 
screenTop = 350; 
screenBottom = 650;
```

## Output

### Video Output
- Annotated MP4 video with:
  - Facial landmarks (68-point mesh)
  - Gaze direction vectors
  - Head pose axes (yaw, pitch, roll indicators)
  - Real-time alerts and anomaly markers
  - Severity color-coding (red for critical, yellow for warnings)

### Anomaly Log
Timestamped records of all detected incidents:
- **Type**: Face absence, gaze deviation, head pose deviation, lighting change, phone, video skip, etc.
- **Severity**: Normal, Warning, Critical
- **Timestamp**: Frame number and elapsed time

### Report
Comprehensive proctoring report generated via `generateReport.m`:
- Summary statistics
- Incident timeline
- Behavioral analysis
- Recommendations for review

## Model Details

### Face Landmarks (ONNX)
- **Model**: Pre-trained ONNX neural network
- **Output**: 68 facial landmark points
- **Used for**: Eye region localization, pupil detection, gaze vector calculation

### Gaze Estimation (L2CS-Net)
- **Architecture**: ResNet50 backbone
- **Input**: Face crop image
- **Output**: Pitch and yaw angles (radians)
- **Advantages**: 360° gaze coverage, robust across head poses

### YOLO-v8 Object Detection
- **Purpose**: General object detection (phones, secondary objects)
- **Weights**: Pre-trained COCO weights included

## Known Limitations & Future Work

- Requires good lighting conditions for optimal performance
- Gaze estimation accuracy decreases with extreme head angles (>45°)
- Requires calibration between different camera setups
- Performance depends on hardware (GPU recommended)

## Troubleshooting

### Python Import Errors
- Ensure Python path is correctly configured in MATLAB
- Verify L2CS-Net is installed: `pip install l2cs-net`
- Check that gaze_engine.py is in the MATLAB working directory

### Video Reader Issues
- Supported formats: MP4, AVI, MOV (depends on codec)
- Ensure codec is installed on system

### Model Loading Errors
- Verify model files exist in `models/` directory
- Check file permissions and paths


## Contributors

- Regindin, Sean Adrien
- Cabrera, James
- Cheung, Tsz
- Llovit, Ben

## References

- **L2CS-Net**: [Paper on Gaze Estimation](https://arxiv.org/abs/2203.06997)
- **YOLO-v8**: [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/)
- **MATLAB Computer Vision**: [MATLAB Documentation](https://www.mathworks.com/help/vision/)

---

**For questions or issues, please open an issue in the repository.**