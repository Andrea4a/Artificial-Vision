# Pedestrian attribute recognition

## Installation

To execute the project, follow these steps:

1. Install the required Python packages by running the following command in your terminal:

   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you have Python installed on your system. Python version 3.7 or higher is recommended.

## Usage

The program analyzes a video file and extracts information based on the provided configuration. Use the following command to execute the code:

```bash
python main.py --video video/video.mp4 --results results/results.txt --configuration configuration/config.txt
```

### Parameters
- `--video`: Path to the video file to be analyzed.
- `--results`: Path to the output file where the analysis results will be saved.
- `--configuration`: Path to the configuration file specifying the directional line coordinates.

### Example
If you have:
- A video file named `video.mp4` in the `video/` folder.
- A configuration file `configuration.txt` in the `configuration/` folder.

Run:

```bash
python main.py --video video/video.mp4 --results results/results.txt --configuration configuration/config.txt
```

### Output
The results will be stored in the file specified in the `--results` parameter, in JSON format, containing all extracted details from the video analysis.

### Additional Notes
Ensure the folders `video/`, `results/`, and `configuration/` exist and contain the appropriate files before executing the command.
