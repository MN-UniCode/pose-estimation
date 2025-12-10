# Multimodal Motion Tracking & Pose Estimation

This project provides a multimodal system for motion tracking and human pose estimation using video input or live camera streaming. It computes the **kinetic energy** of people detected in the scene and visualizes results through histograms. 

Energy metrics are computed for:

* **Whole body**
* **Upper body**
* **Lower body**
* **Left arm**
* **Right arm**
* **Left leg**
* **Right leg**

The system supports both **Mediapipe** (single-person) and **YOLO** (multi-person) pose estimation models.

---

## Features

* Multimodal pose estimation pipeline.
* Support for video files from the [dataset](./project/videos) or live camera streaming.
* Kinetic energy computation for different body regions.
* Histogram visualization of kinetic energy over time.
* Automated environment setup via [setup.sh](./setup.sh).

---

## Installation

Make the setup script executable and run it:

```bash
chmod +x setup.sh
./setup.sh
```

This will:

* Create a Python virtual environment.
* Install all required dependencies.

---

## Requirements

* Python 3.10
* [requirements.txt](./requirements.txt)

All dependencies are automatically installed via [`setup.sh`](./setup.sh).

---

## Usage

### 1. Choose Input Source

Edit the configuration in [`project/main`](./project/main.py) to select:

* A **video** from the dataset, or
* **Live camera streaming** by setting:

```python
live_input = True
```

### 2. Adjust Physical Parameters

You can modify the person's weight and height in [`project/main`](./project/main.py) for more accurate kinetic energy estimation:

```python
total_mass = 67      # Person's mass in kg
sub_height_m = 1.75  # Person's height in meters
```

If using multi-person YOLO tracking, it is recommended to set average weight and average height values.

### 3. Run the Project

Execute the main program:

```bash
python3 project/main
```

When prompted, select the pose estimation backend:

* Press **1** for **Mediapipe** (single-person tracking)
* Press **2** for **YOLO** (multi-person tracking)

#### 3.1 Real-Time Histogram Filtering

During real-time visualization, you can filter the histogram using keyboard shortcuts:
* Press `a` → ALL: shows all 7 categories 
* Press `l` → LIMB: shows only limb categories (left/right arms and legs)
* Press `w` → WHOLE: shows body-level groups (whole, upper, lower)
* Press `q` → QUIT: stop the execution

---

## Output

The system will display:

* The pose estimation visualization.
* Computed kinetic energy values.
* A histogram representing kinetic energy distribution across different body parts.
* A message indicating which limb is moving the most, for each detected person.
