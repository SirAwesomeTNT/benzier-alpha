import subprocess
import json

def openSongAtLocation(filePath):
    # Read the location of the audio file from "songLocation.txt"
    with open(filePath, "r") as file:
        songPath = file.read().strip()
    return songPath

def ffprobe(file_path):
    """
    Run ffprobe command to get media file information.
    """
    command = ['ffprobe', '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', file_path]
    result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = result.communicate()
    
    if err:
        print("Error occurred:", err)
        return None
    else:
        # Decode the output to a string
        output_str = out.decode("utf-8")

        # Parse the JSON output
        media_info = json.loads(output_str)

        return media_info

# Example usage:
songPath = openSongAtLocation("songLocation.txt")
audio_info = ffprobe(songPath)

relevantInfo = {'format_name': '',
                'size': '',
                'bit_rate': '',
                'duration_ts': '',
                'bits_per_raw_sample': '',
                'sample_fmt': '',
                'sample_rate': ''}

if audio_info:
    # Accessing format information
    format_info = audio_info.get("format", {})
    if format_info:
        for key in relevantInfo:
            relevantInfo[key] = format_info.get(key, "")

    # Accessing stream information
    streams = audio_info.get("streams", [])
    for stream in streams:
        for key in relevantInfo:
            if key in stream:
                relevantInfo[key] = stream[key]
        break  # Process only the first stream

else:
    print("FFprobe failed to retrieve media information.")

# Now you can access the values in relevantInfo dictionary
for key in relevantInfo:
    if relevantInfo[key]:
        print(f"{key}: {relevantInfo[key]}")