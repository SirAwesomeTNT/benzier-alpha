import subprocess
import numpy as np
import matplotlib.pyplot as plt
import json

def ffprobe(songPath):
    """
    Run ffprobe command to get media file information.
    
    Args:
    - songPath: Path to the media file
    
    Returns:
    - audioInfo: Dictionary containing media file information
    """
    # Define the ffprobe command to retrieve media file information in JSON format
    command = ['ffprobe', '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', songPath]
    
    # Execute the ffprobe command and capture the output
    result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = result.communicate()
    
    # Check if there was an error during execution
    if err:
        print("Error occurred:", err)
        return None
    else:
        # Decode the output to a string
        output_str = out.decode("utf-8")

        # Parse the JSON output to extract media file information
        rawJsonOutput = json.loads(output_str)

        # Initialize an empty dictionary to store media file information
        audioInfo = {'filename': '',
                    'format_name': '',
                    'size': '',
                    'bit_rate': '',
                    'duration': '',
                    'duration_ts': '',
                    'bits_per_raw_sample': '',
                    'sample_rate': '',
                    'sample_fmt': ''}

        # If the JSON output is not empty, extract format and stream information
        if rawJsonOutput:
            format_info = rawJsonOutput.get("format", {})
            if format_info:
                # Update the audioInfo dictionary with format information
                audioInfo.update(format_info)

            streams = rawJsonOutput.get("streams", [])
            if streams:
                # Update the audioInfo dictionary with stream information from the first stream
                audioInfo.update(streams[0])

    return audioInfo

def printInfoDictionary(audioInfo):
    """
    Print the audio information dictionary and return it as a string.
    
    Args:
    - audioInfo: Dictionary containing media file information
    
    Returns:
    - outputText: String representation of the audio information
    """
    outputText = ""
    for key in audioInfo:
        if audioInfo[key]:
            # Print each key-value pair if the value is not empty
            print(f"{key}: {audioInfo[key]}")
            outputText += f"{key}: {audioInfo[key]}\n"

    return outputText

def openSongAtFileLocation(filePath):
    """
    Read the location of the audio file from a text file.
    
    Args:
    - filePath: Path to the text file containing the location of the audio file
    
    Returns:
    - songPath: Path to the audio file
    """
    with open(filePath, "r") as file:
        songPath = file.read().strip()

    return songPath

def writeOutput(outputText):
    """
    Write the output text to a file.
    
    Args:
    - outputText: Text to be written to the file
    """
    with open("output.txt", "w") as output_file:
        output_file.write(outputText)

def extractSamples(file_path):
    """
    Extract left and right channel samples from the audio file.
    
    Args:
    - file_path: Path to the audio file
    
    Returns:
    - l_samples: Left channel samples as a NumPy array
    - r_samples: Right channel samples as a NumPy array
    """
    try:
        # Execute FFmpeg command to extract raw PCM audio samples
        command = ['ffmpeg', '-i', file_path, '-vn', '-f', 'f32le', '-']
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_audio_data = result.stdout
        
        # Convert raw audio data to NumPy array of float32 type
        audio_np = np.frombuffer(raw_audio_data, dtype='float32')
        
        # Separate left and right channel samples
        l_samples = audio_np[::2]  # Every other sample starting from index 0 (left channel)
        r_samples = audio_np[1::2]  # Every other sample starting from index 1 (right channel)
        
        return l_samples, r_samples
    except subprocess.CalledProcessError as e:
        # Handle errors during subprocess execution
        print("Error:", e.stderr)
        return None, None

# Open the audio file location from a text file
songPath = openSongAtFileLocation("bezier-alpha/songLocation.txt")

# Extract left and right channel samples from the audio file
l, r = extractSamples(songPath)
print("Left Channel Samples:", l[:20])
print("Right Channel Samples:", r[:20])

# Write the audio information to an output file
writeOutput(printInfoDictionary(ffprobe(songPath)))

# Generate time axis based on the number of samples and sample rate
num_samples = len(l)  # Assuming l and r have the same length
sample_rate = 44100  # Assuming a sample rate of 44100 Hz
duration = num_samples / sample_rate
time_axis = np.linspace(0, duration, num_samples)

# Plot left and right channel samples (only the first 20 samples)
plt.figure(figsize=(10, 6))
plt.plot(time_axis[:20:], l[:20:], label='Left Channel')
plt.plot(time_axis[:20:], r[:20:], label='Right Channel')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Left and Right Channel Samples (First 20 Samples)')
plt.legend()
plt.grid(True)
plt.show()