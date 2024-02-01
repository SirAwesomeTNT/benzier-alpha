import subprocess

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

def ffmpeg(songPath, command):
    """
    Run FFmpeg command directly on the audio file specified in "songLocation.txt".
    """
    try:
        # Execute the FFmpeg command
        result = subprocess.run(command + [songPath], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return None, e.stderr

def openSongAtLocation(filePath):
    # Read the location of the audio file from "songLocation.txt"
    with open(filePath, "r") as file:
        songPath = file.read().strip()

    return songPath

def writeOutput(outputText):
    with open("output.txt", "w") as output_file:
        output_file.write(outputText)

songPath = openSongAtLocation("songLocation.txt")

outputText = ffprobe(songPath)

writeOutput(outputText)

print(outputText)