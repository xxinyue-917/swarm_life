#!/usr/bin/env python3
"""
Script to automatically run particle life simulation and save videos for all presets.
"""

# ============== CONFIGURATION ==============
# Modify these values to change the script behavior

PRESETS_DIR = 'presets'      # Directory containing preset JSON files
OUTPUT_DIR = 'videos'         # Directory to save videos
VIDEO_DURATION = 10           # Video duration in seconds
FPS = 30                      # Frames per second
PROCESS_SINGLE = None         # Set to a file path to process single preset, None for all

# ============================================

import os
import sys
import glob
import argparse
import numpy as np
import pygame
import cv2
from pathlib import Path

# Add parent directory to path to import particle_life
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from particle_life import Config, ParticleLife


class VideoRecorder:
    """Records pygame surface to video file"""

    def __init__(self, output_path, fps=30, width=1000, height=1000):
        """Initialize video recorder with OpenCV"""
        self.fps = fps
        self.width = width
        self.height = height

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not self.out.isOpened():
            raise ValueError(f"Failed to open video writer for {output_path}")

    def add_frame(self, surface):
        """Add a pygame surface as a frame to the video"""
        # Convert pygame surface to numpy array
        frame = pygame.surfarray.array3d(surface)
        # Rotate and flip to correct orientation
        frame = np.rot90(frame)
        frame = np.flipud(frame)
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Write frame
        self.out.write(frame)

    def release(self):
        """Release the video writer"""
        self.out.release()


class SimulationVideoSaver:
    """Handles running simulation and saving videos"""

    def __init__(self, video_duration=10, fps=30):
        """
        Initialize video saver

        Args:
            video_duration: Duration of video in seconds
            fps: Frames per second for video
        """
        self.video_duration = video_duration
        self.fps = fps
        self.total_frames = video_duration * fps

        # Initialize pygame
        pygame.init()

    def run_and_save(self, config_path, output_path):
        """
        Run simulation with given config and save video

        Args:
            config_path: Path to configuration JSON file
            output_path: Path where video will be saved
        """
        print(f"\nProcessing: {config_path}")
        print(f"Output: {output_path}")

        # Load configuration
        config = Config.load(config_path)

        # Create simulation
        sim = ParticleLife(config)

        # Create video recorder
        recorder = VideoRecorder(output_path, self.fps, config.width, config.height)

        # Run simulation for specified duration
        clock = pygame.time.Clock()
        frame_count = 0

        print(f"Recording {self.video_duration} seconds ({self.total_frames} frames)...")

        while frame_count < self.total_frames:
            # Handle pygame events to prevent hanging
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break

            # Step simulation
            sim.step()

            # Clear screen
            sim.screen.fill((255, 255, 255))  # White background

            # Draw particles
            for i in range(sim.n):
                pos = sim.positions[i]
                species = sim.species[i]
                color = sim.colors[species]

                # Draw particle
                x = int(pos[0])
                y = int(pos[1])

                if sim.show_orientations:
                    # Draw as circle with orientation line
                    angle = sim.orientations[i]
                    radius = 5

                    # Draw circle
                    pygame.draw.circle(sim.screen, color, (x, y), radius)

                    # Draw orientation line
                    line_length = radius * 0.8
                    end_x = x + line_length * np.cos(angle)
                    end_y = y + line_length * np.sin(angle)
                    pygame.draw.line(sim.screen, (0, 0, 0), (x, y), (end_x, end_y), 1)
                else:
                    # Draw simple circle
                    pygame.draw.circle(sim.screen, color, (x, y), 4)

            # Add frame to video
            recorder.add_frame(sim.screen)

            # Update display (optional, for monitoring)
            pygame.display.flip()

            # Control frame rate
            clock.tick(self.fps)

            frame_count += 1

            # Progress indicator
            if frame_count % self.fps == 0:
                seconds = frame_count // self.fps
                print(f"  {seconds}/{self.video_duration} seconds recorded")

        # Clean up
        recorder.release()
        pygame.quit()
        pygame.init()  # Reinitialize for next video

        print(f" Video saved: {output_path}")

    def process_all_presets(self, presets_dir="presets", output_dir="videos"):
        """
        Process all preset files in the presets directory

        Args:
            presets_dir: Directory containing preset JSON files
            output_dir: Directory where videos will be saved
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Find all preset files
        preset_files = glob.glob(os.path.join(presets_dir, "*.json"))

        if not preset_files:
            print(f"No preset files found in {presets_dir}")
            return

        print(f"Found {len(preset_files)} preset(s) to process")

        # Process each preset
        for i, preset_path in enumerate(preset_files, 1):
            # Get preset filename without extension
            preset_name = Path(preset_path).stem

            # Create output filename with timestamp
            output_filename = f"{preset_name}.mp4"
            output_path = os.path.join(output_dir, output_filename)

            print(f"\n[{i}/{len(preset_files)}] Processing preset: {preset_name}")

            try:
                self.run_and_save(preset_path, output_path)
            except Exception as e:
                print(f" Error processing {preset_name}: {e}")
                continue

        print(f"\n All videos saved to {output_dir}/")


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate videos from particle life presets')
    parser.add_argument('--load', type=str, help='Path to specific preset file to process')
    args = parser.parse_args()

    # Override PROCESS_SINGLE if --load argument is provided
    process_single = args.load if args.load else PROCESS_SINGLE

    print("=" * 50)
    print("Particle Life Video Generator")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Presets directory: {PRESETS_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Video duration: {VIDEO_DURATION} seconds")
    print(f"  FPS: {FPS}")
    print(f"  Single file mode: {process_single if process_single else 'No (processing all presets)'}")
    print("=" * 50)

    # Create video saver
    saver = SimulationVideoSaver(video_duration=VIDEO_DURATION, fps=FPS)

    if process_single:
        # Process single file
        if not os.path.exists(process_single):
            print(f"Error: File {process_single} not found")
            sys.exit(1)

        # Create output directory if needed
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Generate output filename
        preset_name = Path(process_single).stem
        output_path = os.path.join(OUTPUT_DIR, f"{preset_name}.mp4")

        # Process the file
        saver.run_and_save(process_single, output_path)
    else:
        # Process all presets in directory
        saver.process_all_presets(PRESETS_DIR, OUTPUT_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()