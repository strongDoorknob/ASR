import os
import subprocess
import sys

# Try to import tqdm, but fallback gracefully if not installed
try:
    from tqdm import tqdm
except ImportError:
    print("❌ 'tqdm' library not found. Please run: pip install tqdm")
    sys.exit(1)

def main(folder_path):
    # 1. Setup paths
    # We use absolute paths to prevent "File not found" errors in FFmpeg
    folder_path = os.path.abspath(folder_path)
    list_file_path = os.path.join(folder_path, "mylist.txt")
    output_audio_path = os.path.join(folder_path, "combined_audio.m4a")
    
    # 2. Find and sort files
    if not os.path.exists(folder_path):
        print(f"❌ Error: The folder '{folder_path}' does not exist.")
        return

    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.m4a')]
    files.sort()

    if not files:
        print(f"❌ No .m4a files found in '{folder_path}'")
        return

    print(f"📂 Found {len(files)} audio files in '{folder_path}'")

    # 3. Write the list file (with Progress Bar)
    # TQDM wraps the loop to show a progress bar while writing the text file
    with open(list_file_path, "w", encoding="utf-8") as f:
        for filename in tqdm(files, desc="Preparing file list", unit="file"):
            abs_path = os.path.join(folder_path, filename)
            # FFmpeg requires safe quoting for paths with spaces/apostrophes
            safe_path = abs_path.replace("'", "'\\''") 
            f.write(f"file '{safe_path}'\n")

    print("🚀 Starting FFmpeg merge... (This might take a moment)")

    # 4. Run FFmpeg
    # We allow FFmpeg to print its own progress to the console so you can see it working
    command = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file_path,
        "-c", "copy",  # Fast copy mode (no quality loss)
        output_audio_path,
        "-y"           # Overwrite output if exists
    ]

    try:
        subprocess.run(command, check=True)
        print(f"\n✅ Success! Combined file saved at:\n{output_audio_path}")
        
        # Cleanup: Remove the temporary text file
        if os.path.exists(list_file_path):
            os.remove(list_file_path)
            
    except FileNotFoundError:
        print("\n❌ Error: 'ffmpeg' command not found. Install it (brew install ffmpeg) and try again.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FFmpeg failed to merge files. Error code: {e.returncode}")

if __name__ == "__main__":
    # Change this to your actual folder path
    folder_to_combine = 'combine/' 
    main(folder_to_combine)