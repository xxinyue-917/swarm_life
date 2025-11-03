# Video Recording for Particle Life Simulation

This script automatically generates videos from saved configuration presets.

## Installation

First, install the required dependencies:

```bash
pip install -r requirements_video.txt
```

Or install OpenCV directly:

```bash
pip install opencv-python
```

## Usage

### Basic Usage

Simply run the script to process all presets:

```bash
python src/save_videos.py
```

This will:
- Load each `.json` file from `presets/`
- Run the simulation for 10 seconds (by default)
- Save videos to `videos/` directory

### Command-Line Usage

Process a specific preset using the `--load` argument:

```bash
python src/save_videos.py --load presets/2_move_together.json
```

This overrides the `PROCESS_SINGLE` configuration and processes only the specified file.

### Configuration

All settings are configured at the top of the `save_videos.py` file. Edit these values to customize behavior:

```python
# ============== CONFIGURATION ==============
PRESETS_DIR = 'presets'      # Directory containing preset JSON files
OUTPUT_DIR = 'videos'         # Directory to save videos
VIDEO_DURATION = 10           # Video duration in seconds
FPS = 30                      # Frames per second
PROCESS_SINGLE = None         # Set to a file path to process single preset, None for all
# ============================================
```

### Examples

**Process all presets with default settings:**
```bash
python src/save_videos.py
```

**Process a single preset file (via command line):**
```bash
python src/save_videos.py --load presets/config_20241102_143022.json
```

**Process a single preset file (via configuration):**
Edit the script and change:
```python
PROCESS_SINGLE = 'presets/config_20241102_143022.json'
```

**Generate longer videos:**
Edit the script and change:
```python
VIDEO_DURATION = 20  # 20 second videos
```

**Use different directories:**
Edit the script and change:
```python
PRESETS_DIR = 'my_presets'
OUTPUT_DIR = 'my_videos'
```

## Output

Videos are saved as MP4 files in the `videos/` directory with names matching the preset files:
- `config_20241102_143022.json` → `config_20241102_143022.mp4`

## Tips

1. **Create interesting presets first**: Use the main simulation (`python src/particle_life.py`) to create and save interesting configurations with the 'S' key.

2. **Batch processing**: The script will process all presets automatically, showing progress for each video.

3. **Video quality**: Higher FPS values create smoother videos but larger file sizes.

4. **Performance**: Video generation runs at real-time speed (10 seconds of video takes ~10 seconds to generate).

## Example Workflow

1. Run the simulation and create interesting patterns:
   ```bash
   python src/particle_life.py
   ```
   - Edit matrices with 'M'
   - Save configurations with 'S'

2. Generate videos for all saved configurations:
   ```bash
   python src/save_videos.py
   ```

3. Find your videos in the `videos/` directory!