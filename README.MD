# KC Agent - Feature Matching and Automation Tools

This project provides a suite of tools for feature matching, window automation, and GUI interaction using computer vision techniques.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Scripts Documentation](#scripts-documentation)
  - [idc_find_feature.py](#idc_find_featurepy)
  - [act_mouse_movement.py](#act_mouse_movementpy)
  - [flow_setting.py](#flow_settingpy)
  - [kc_simulator.py](#kc_simulatorpy)
  - [logger.py](#loggerpy)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Project Overview

The KC Agent project consists of several Python scripts that work together to:
- Find and match visual features in windows using computer vision
- Automate mouse movements and clicks
- Create and manage GUI workflows
- Simulate application interfaces for testing

## Installation

### Prerequisites

- Python 3.8 or higher
- Windows OS (required for win32 API calls)
- Virtual environment recommended

### Dependencies

```bash
pip install opencv-python
pip install numpy
pip install mss
pip install pywin32
pip install pyside6
pip install pathlib
```

### Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies
4. Create necessary directories:
   ```
   kc_agent/
   ├── screenshots/     # Store application screenshots
   ├── crops/          # Store cropped template images
   └── logs/           # Log files (created automatically)
   ```

## Scripts Documentation

### idc_find_feature.py

**Purpose**: Core feature matching script that finds template images within window captures using computer vision techniques.

#### Features
- Two matching methods: ORB feature matching and template matching
- Returns structured MatchResult objects with coordinates
- Visual debugging with overlay display
- Logging support

#### Usage

```python
# Programmatic usage
from idc_find_feature import FeatureMatcher

matcher = FeatureMatcher("Window Title", "path/to/template.png")
result = matcher.match_template_ORB(min_matches=20, ratio=0.75)
# or
result = matcher.match_template_tm(threshold=0.8)

if result.found:
    print(f"Found at screen coordinates: {result.bbox_screen_xywh}")
    matcher.visualize_match(result)
```

```bash
# Direct execution (uses hardcoded values)
python idc_find_feature.py
```

#### MatchResult Structure
```python
@dataclass
class MatchResult:
    found: bool                              # Whether match was found
    score: float                            # Match confidence score
    bbox_xywh: Tuple[int, int, int, int]   # Client area coordinates (x, y, w, h)
    bbox_screen_xywh: Tuple[int, int, int, int]  # Screen coordinates (x, y, w, h)
    roi: Tuple[int, int, int, int]          # Region of interest
    method: str                             # Matching method used
```

#### Methods
- `match_template_ORB()`: Feature-based matching using ORB descriptors
- `match_template_tm()`: Template matching using OpenCV
- `visualize_match()`: Display match visualization
- `capture_window()`: Capture window screenshot

---

### act_mouse_movement.py

**Purpose**: Utility for controlling mouse cursor position and movements on screen.

#### Features
- Move mouse to specific screen coordinates
- Support for relative positioning within rectangles
- Logging of mouse movements

#### Usage

```python
from act_mouse_movement import move_mouse_to_rect

# Move to center of rectangle (x, y, width, height)
move_mouse_to_rect((100, 200, 300, 400))

# Move to specific relative position (30% from left, 70% from top)
move_mouse_to_rect((100, 200, 300, 400), relative_x=0.3, relative_y=0.7)
```

```bash
# Direct execution (uses hardcoded coordinates)
python act_mouse_movement.py
```

#### Parameters
- `rect`: Tuple of (x, y, width, height)
- `relative_x`: X position relative to rectangle width (0.0 to 1.0)
- `relative_y`: Y position relative to rectangle height (0.0 to 1.0)

---

### flow_setting.py

**Purpose**: GUI tool for creating and managing image regions for automation workflows.

#### Features
- Interactive region annotation on images
- Automatic image discovery from screenshots folder
- Export regions to JSON configuration
- Crop generation for templates

#### Usage

```bash
# Launch GUI
python flow_setting.py
```

#### GUI Controls
- **Load Images**: Automatically discovers images from `screenshots/` folder
- **Add Region**: Click and drag on image to create regions
- **Export**: Saves regions to `regions_map.json`
- **Generate Crops**: Creates cropped template images in `crops/` folder

#### Output Format
```json
{
  "screenshots/image1.png": [
    {
      "name": "crops/image1/region1.png",
      "rect": [x, y, width, height]
    }
  ]
}
```

---

### kc_simulator.py

**Purpose**: GUI simulator that displays images and manages region interactions for testing automation workflows.

#### Features
- Load and display images from configuration
- Interactive region detection
- Dynamic image switching based on region clicks
- Window management for automation testing

#### Usage

```bash
# Run simulator
python kc_simulator.py
```

#### Configuration
Uses `regions_map.json` to define:
- Base images to display
- Interactive regions
- Image transitions on region clicks

---

### logger.py

**Purpose**: Centralized logging configuration for all scripts in the project.

#### Features
- Dual output: Console and file logging
- Timestamped log entries
- Automatic log directory creation
- Configurable log levels

#### Usage

```python
from logger import setup_logger

# Setup logger with custom name
logger = setup_logger("my_script")

# Use logger
logger.info("Information message")
logger.debug("Debug message")
logger.warning("Warning message")
logger.error("Error message")
```

#### Log Format
```
[logger_name] - YYYY-MM-DD HH:MM:SS,mmm - LEVEL - message
```

#### Log Files
- Location: `logs/kc_agent_YYYYMMDD_HHMMSS.log`
- Automatic rotation with timestamps

## Usage Examples

### Complete Automation Workflow

```python
from idc_find_feature import FeatureMatcher
from act_mouse_movement import move_mouse_to_rect

# 1. Find a button on screen
matcher = FeatureMatcher("Application Window", "templates/button.png")
result = matcher.match_template_tm(threshold=0.8)

if result.found:
    # 2. Move mouse to the button center
    move_mouse_to_rect(result.bbox_screen_xywh)

    # 3. Click (requires additional implementation)
    # click_at_position(result.bbox_screen_xywh)
```

### Batch Processing

```python
import json
from idc_find_feature import FeatureMatcher

# Load regions configuration
with open('regions_map.json', 'r') as f:
    regions = json.load(f)

# Check each region
for image_path, region_list in regions.items():
    matcher = FeatureMatcher("Window Title", image_path)
    for region in region_list:
        result = matcher.match_template_tm(threshold=0.8)
        print(f"Region {region['name']}: {'Found' if result.found else 'Not found'}")
```

## Configuration

### Environment Variables
- `LOG_LEVEL`: Set default logging level (DEBUG, INFO, WARNING, ERROR)

### File Structure
```
kc_agent/
├── screenshots/          # Application screenshots
├── crops/               # Template crops
├── logs/                # Log files (auto-created)
├── regions_map.json     # Region configuration
├── idc_find_feature.py  # Core matching script
├── act_mouse_movement.py # Mouse control
├── flow_setting.py      # GUI region editor
├── kc_simulator.py      # GUI simulator
└── logger.py           # Logging configuration
```

## Troubleshooting

### Common Issues

1. **Window not found**
   - Ensure target window is open and visible
   - Check window title substring matches exactly
   - Verify window is not minimized

2. **Template not found**
   - Check template image path is correct
   - Ensure template image format is supported (PNG recommended)
   - Try adjusting threshold/match parameters

3. **Mouse movement not working**
   - Ensure script has necessary permissions
   - Check if screen coordinates are within monitor bounds
   - Verify DPI awareness settings

4. **Logging issues**
   - Ensure write permissions in project directory
   - Check available disk space
   - Verify log directory creation

### Debug Tips

1. Enable visualization to see matching results:
   ```python
   matcher.visualize_match(result)
   ```

2. Use debug logging:
   ```python
   logger = setup_logger("debug", logging.DEBUG)
   ```

3. Test with different matching methods:
   - ORB for feature-rich images
   - Template matching for exact duplicates

## License

[Add your license information here]