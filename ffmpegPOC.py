import subprocess
import numpy as np
import matplotlib.pyplot as plt
import json

def ffprobe(songPath):
    """
    Run ffprobe command to get media file information.
    """
    command = ['ffprobe', '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', songPath]
    result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = result.communicate()
    
    if err:
        print("Error occurred:", err)
        return None
    else:
        # Decode the output to a string
        output_str = out.decode("utf-8")

        # Parse the JSON output
        rawJsonOutput = json.loads(output_str)

        audioInfo = {'filename': '',
                    'format_name': '',
                    'size': '',
                    'bit_rate': '',
                    'duration': '',
                    'duration_ts': '',
                    'bits_per_raw_sample': '',
                    'sample_rate': '',
                    'sample_fmt': ''}

    if rawJsonOutput:
        # Accessing format information
        format_info = rawJsonOutput.get("format", {})
        if format_info:
            for key in audioInfo:
                audioInfo[key] = format_info.get(key, "")

        # Accessing stream information
        streams = rawJsonOutput.get("streams", [])
        for stream in streams:
            for key in audioInfo:
                if key in stream:
                    audioInfo[key] = stream[key]
            break  # Process only the first stream

    else:
        print("FFprobe failed to retrieve media information.")

    return audioInfo

def printInfoDictionary(audioInfo):
    outputText = ""
    for key in audioInfo:
        if audioInfo[key]:
            print(f"{key}: {audioInfo[key]}")
            outputText += f"{key}: {audioInfo[key]}\n"

    return outputText

def openSongAtLocation(filePath):
    # Read the location of the audio file from "songLocation.txt"
    with open(filePath, "r") as file:
        songPath = file.read().strip()

    return songPath

def writeOutput(outputText):
    with open("output.txt", "w") as output_file:
        output_file.write(outputText)

def extractSamples(file_path):
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
        print("Error:", e.stderr)
        return None, None

songPath = openSongAtLocation("songLocation.txt")

l, r = extractSamples(songPath)
print("Left Channel Samples:", l[:20])
print("Right Channel Samples:", r[:20])

writeOutput(printInfoDictionary(ffprobe(songPath)))

# Generate time axis based on the number of samples and sample rate
num_samples = len(l)  # Assuming l and r have the same length
sample_rate = 44100  # Assuming a sample rate of 44100 Hz
duration = num_samples / sample_rate
time_axis = np.linspace(0, duration, num_samples)

# Plot left and right channel samples
plt.figure(figsize=(10, 6))

plt.plot(time_axis[::10], l[::10], label='Left Channel')
plt.plot(time_axis[::10], r[::10], label='Right Channel')

plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Left and Right Channel Samples')
plt.legend()
plt.grid(True)
plt.show()
