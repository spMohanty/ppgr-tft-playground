#!/usr/bin/env python3
import os
import re
import argparse
import imageio.v2 as imageio

def generate_animations(folder, fps, out_format, output_dir):
    """
    Scans the given folder for PNG files following the naming convention:
      {plot-type-name}_{step_number}_{hash}.png
    Groups the files by the plot-type, orders them by the step number,
    and generates an animation (GIF or video) for each plot type.
    
    Parameters:
      folder (str): Path to the folder with the images.
      fps (float): Frames per second for the animation.
      out_format (str): Output format, e.g., 'gif' or 'mp4'.
      output_dir (str): Directory where animations will be saved.
    """
    # Regular expression to capture plot_type, step, and hash.
    # This assumes the plot_type may contain underscores.
    pattern = re.compile(r'^(?P<plot_type>.+)_(?P<step>\d+)_(?P<hash>[0-9a-f]+)\.png$')

    # Dictionary to collect images for each plot type.
    animations = {}

    # List all files in the folder and process those matching our pattern.
    for filename in os.listdir(folder):
        if not filename.endswith('.png'):
            continue
        match = pattern.match(filename)
        if not match:
            # Skip files that don't match the naming convention.
            continue
        plot_type = match.group('plot_type')
        step = int(match.group('step'))
        filepath = os.path.join(folder, filename)
        animations.setdefault(plot_type, []).append((step, filepath))

    # Process each group: sort by step number and generate the animation.
    for plot_type, entries in animations.items():
        # Sort entries by the step number.
        sorted_entries = sorted(entries, key=lambda x: x[0])
        
        # Read images into a list.
        frames = []
        for step, filepath in sorted_entries:
            try:
                frame = imageio.imread(filepath)
                frames.append(frame)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        
        if not frames:
            print(f"No frames found for plot type {plot_type}.")
            continue
        
        # Define output file path.
        out_filename = f"{plot_type}_animation.{out_format}"
        out_path = os.path.join(output_dir, out_filename)
        
        print(f"Saving animation for '{plot_type}' with {len(frames)} frames to {out_path}")
        try:
            if out_format.lower() == "gif":
                # Save as GIF.
                imageio.mimsave(out_path, frames, fps=fps)
            elif out_format.lower() in ["mp4", "avi"]:
                # Save as video (requires imageio-ffmpeg installed).
                writer = imageio.get_writer(out_path, fps=fps)
                for frame in frames:
                    writer.append_data(frame)
                writer.close()
            else:
                print(f"Unsupported output format: {out_format}")
        except Exception as e:
            print(f"Error saving animation for {plot_type}: {e}")

def main():
    folder = "/scratch/mohanty/wandb/wandb/run-20250207_115725-saewdb31/files/media/images/"
    fps = 5.0
    out_format = "gif"
    output_dir = "animations/"

    # Use the input folder as output if none is provided.
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    generate_animations(folder, fps, out_format, output_dir)

if __name__ == "__main__":
    main()
