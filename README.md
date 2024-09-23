# annotation-tool
Tool to automatically annotate video files 

This tool allows you to annotate objects in a video based on a specified prompt. The annotations are saved in the desired format at a specified location. Below are the command-line arguments available for this tool.

## Usage

```bash
python annotation_tool.py <input_path> <prompt> <save_path> [--nframe N] [--only_txt True]

1. **`input_path`** (type: `str`):  
   Path to the video file.
   
2. **`prompt`** (type: `str`):  
   The prompt specifying what objects to detect.
   
3. **`save_path`** (type: `str`):  
   Path where the annotations will be saved.

1. **`--nframe`** (type: `int`):  
   Specifies which frames to leave. For example, if `2` is specified, every 2nd frame will be annotated.
   
2. **`--only_txt`** (type: `bool`):  
   Set to `True` if only `.txt` files are needed.