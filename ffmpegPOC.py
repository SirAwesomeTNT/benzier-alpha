import subprocess
import numpy as np

def ffprobe(songPath):
    """
    Run ffprobe command to get media file information.
    """
    command = ['ffprobe', '-v', 'error', '-print_format', 'flat', '-show_format', '-show_streams', songPath]
    result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = result.communicate()
    
    relevant_info = ""

    if err:
        print("Error occurred:", err)
    else:
        # Decode the output to a string
        output_str = out.decode("utf-8")

        # Filter relevant information
        for line in output_str.split('\n'):
            if not line.startswith("format.tag") and not line.startswith("streams.stream.1"):
                relevant_info += line + '\n'

    return relevant_info

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
        command = ['ffmpeg', '-i', file_path, '-vn', '-af', 'pan=stereo|c0=c0|c1=c1', '-f', 'f32le', '-']
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_audio_data = result.stdout
        
        # Convert raw audio data to NumPy array of int16 type
        audio_np = np.frombuffer(raw_audio_data, dtype='float32')
        
        # Separate left and right channel samples
        l_samples = audio_np[::2]  # Every other sample starting from index 0 (left channel)
        r_samples = audio_np[1::2]  # Every other sample starting from index 1 (right channel)
        
        return l_samples, r_samples
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        return None, None

songPath = openSongAtLocation("songLocation.txt")

outputText = ffprobe(songPath)

l, r = extractSamples(songPath)
print("Left Channel Samples:", l[:20])
print("Right Channel Samples:", r[:20])

writeOutput(outputText)

print(outputText)