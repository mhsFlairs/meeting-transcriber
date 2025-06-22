import os
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





# file_path = "C:\\Users\\mohamed.shokry\\Downloads\\AIMY phone, what's next-20250620_195742-Meeting Recording.mp4"
# audio_file_path = convert_video_to_audio_cached(
#     "example_file_hash_1234567890abcdef", file_path
# )["audio_path"]
# print(f"Audio file path: {audio_file_path}")
file_hash = "example_file_hash_1234567890abcdef"
result = chunk_audio_file(file_hash, "C:\\Users\\MOHAME~1.SHO\\AppData\\Local\\Temp\\tmpwnvxdqfp.mp3", chunk_size_mb=5)
print(result)
