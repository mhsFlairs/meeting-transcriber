import streamlit as st
import tempfile
import os
import hashlib
from openai import AzureOpenAI
from moviepy import VideoFileClip
from pathlib import Path
import time
from dotenv import load_dotenv

# Load environment variables from .env file if it exists

# Load environment variables
load_dotenv(override=True)


# Initialize session state
if "processed_audio_path" not in st.session_state:
    st.session_state.processed_audio_path = None
if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None
if "conversion_complete" not in st.session_state:
    st.session_state.conversion_complete = False
if "uploaded_file_info" not in st.session_state:
    st.session_state.uploaded_file_info = None
if "transcription_complete" not in st.session_state:
    st.session_state.transcription_complete = False
if "transcription_text" not in st.session_state:
    st.session_state.transcription_text = ""
if "translation_complete" not in st.session_state:
    st.session_state.translation_complete = False
if "translation_text" not in st.session_state:
    st.session_state.translation_text = ""


# Initialize OpenAI client
@st.cache_resource
def get_openai_client(api_version) -> AzureOpenAI:
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=api_version,
    )
    return client


def get_file_hash(file_bytes):
    """Generate hash for file content to detect changes"""
    return hashlib.md5(file_bytes).hexdigest()


@st.cache_data(show_spinner=False)
def get_file_info(_file_bytes, filename, file_hash):
    """Get detailed information about the file - cached based on content hash"""
    # Create temp file from bytes
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(filename).suffix
    ) as tmp_file:
        tmp_file.write(_file_bytes)
        temp_path = tmp_file.name

    is_video = is_video_file(filename)
    file_info = {
        "size_mb": len(_file_bytes) / (1024 * 1024),
        "format": Path(filename).suffix.lower(),
        "temp_path": temp_path,
        "name": filename,
        "is_video": is_video,
        "is_audio": is_audio_file(filename),
        "hash": file_hash,
    }

    if is_video:
        try:
            with VideoFileClip(temp_path) as video:
                file_info.update(
                    {
                        "duration": video.duration,
                        "fps": video.fps,
                        "resolution": (
                            f"{video.w}x{video.h}" if video.w and video.h else "Unknown"
                        ),
                        "has_audio": video.audio is not None,
                    }
                )
        except Exception as e:
            st.warning(f"Could not read video metadata: {str(e)}")

    return file_info


@st.cache_data(show_spinner=False)
def convert_video_to_audio_cached(file_hash, temp_video_path):
    """Convert video file to audio - cached based on file hash"""
    try:
        video = VideoFileClip(temp_video_path)

        if video.audio is None:
            video.close()
            raise ValueError("No audio track found in the video file")

        # Create temp audio file
        temp_audio_path = tempfile.mktemp(suffix=".wav")

        # Extract audio
        audio = video.audio
        audio.write_audiofile(temp_audio_path, logger=None)

        # Cleanup
        video.close()
        audio.close()

        # Get audio file info
        audio_size = os.path.getsize(temp_audio_path) / (1024 * 1024)

        return {
            "audio_path": temp_audio_path,
            "size_mb": audio_size,
            "success": True,
            "message": f"Audio extracted successfully! ({audio_size:.1f} MB)",
        }

    except Exception as e:
        return {
            "audio_path": None,
            "success": False,
            "message": f"Error extracting audio: {str(e)}",
        }


def stream_transcription_real(client: AzureOpenAI, audio_file_path, language=None):
    """Real streaming transcription using gpt-4o-mini-transcribe"""
    try:
        with open(audio_file_path, "rb") as audio_file:
            stream = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
                response_format="text",
                language=language,
                stream=True,
            )

            for event in stream:
                print(f"Event {event}")  # Debugging output
                # Handle TranscriptionTextDeltaEvent objects
                if hasattr(event, "delta") and event.delta:
                    yield event.delta
                elif hasattr(event, "text") and event.text:
                    yield event.text
                elif isinstance(event, str):
                    yield event
                # Skip events without text content (like done events)

    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        yield f"Error: {str(e)}"


@st.cache_data(show_spinner=False)
def translate_text_cached(
    text_hash, text, _client: AzureOpenAI, target_language="English"
):
    """Translate text - cached based on text hash and target language"""
    try:
        response = _client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": f"Translate the following text to {target_language}. Only return the translation, no additional text.",
                },
                {"role": "user", "content": text},
            ],
        )
        return {
            "success": True,
            "text": response.choices[0].message.content,
            "error": None,
        }
    except Exception as e:
        return {"success": False, "text": "", "error": str(e)}


def is_video_file(filename):
    """Check if file is a video file"""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}
    return Path(filename).suffix.lower() in video_extensions


def is_audio_file(filename):
    """Check if file is a video file"""
    audio_extensions = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma"}
    return Path(filename).suffix.lower() in audio_extensions


def stream_transcription(text, placeholder, label="Live Transcription"):
    """Simulate streaming text for better UX"""
    words = text.split()
    displayed_text = ""

    for i, word in enumerate(words):
        if i == 0:
            displayed_text = word
        else:
            displayed_text += " " + word

        placeholder.text_area(
            label,
            value=displayed_text,
            height=200,
            key=f"{label.lower().replace(' ', '_')}_{i}",
        )
        time.sleep(0.05)  # Small delay for streaming effect


def stream_translation(text, placeholder, target_language):
    """Simulate streaming translation for better UX"""
    words = text.split()
    displayed_text = ""

    for i, word in enumerate(words):
        if i == 0:
            displayed_text = word
        else:
            displayed_text += " " + word

        placeholder.text_area(
            f"Live Translation ({target_language})",
            value=displayed_text,
            height=200,
            key=f"translation_{target_language.lower()}_{i}",
        )
        time.sleep(0.05)


def upload_file_with_progress(uploaded_file):
    """Upload file with real-time progress tracking"""
    if not uploaded_file:
        return None, None
    
    # Get total file size
    uploaded_file.seek(0, 2)  # Seek to end
    total_size = uploaded_file.tell()
    uploaded_file.seek(0)  # Reset to beginning
    
    # Create progress containers
    progress_container = st.container()
    
    with progress_container:
        st.subheader("📤 File Upload Progress")
        progress_bar = st.progress(0)
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            size_text = st.empty()
            speed_text = st.empty()
        
        with status_col2:
            percent_text = st.empty()
            eta_text = st.empty()
    
    # Read file in chunks with progress tracking
    chunk_size = 1024 * 1024  # 1MB chunks
    file_bytes = b""
    bytes_read = 0
    start_time = time.time()
    
    while True:
        chunk = uploaded_file.read(chunk_size)
        if not chunk:
            break
            
        file_bytes += chunk
        bytes_read += len(chunk)
        
        # Calculate progress
        progress = bytes_read / total_size
        elapsed_time = time.time() - start_time
        
        # Calculate upload speed and ETA
        if elapsed_time > 0:
            speed_mbps = (bytes_read / (1024 * 1024)) / elapsed_time
            remaining_bytes = total_size - bytes_read
            eta_seconds = remaining_bytes / (speed_mbps * 1024 * 1024) if speed_mbps > 0 else 0
        else:
            speed_mbps = 0
            eta_seconds = 0
        
        # Update progress display
        progress_bar.progress(progress)
        
        size_text.text(f"📊 {bytes_read / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
        speed_text.text(f"⚡ Speed: {speed_mbps:.1f} MB/s")
        percent_text.text(f"📈 Progress: {progress * 100:.1f}%")
        
        if eta_seconds > 0:
            eta_text.text(f"⏱️ ETA: {eta_seconds:.1f}s")
        else:
            eta_text.text("⏱️ ETA: Calculating...")
        
        # Small delay to make progress visible for smaller files
        if total_size < 10 * 1024 * 1024:  # Files smaller than 10MB
            time.sleep(0.1)
    
    # Final update
    progress_bar.progress(1.0)
    size_text.text(f"✅ Upload Complete: {total_size / (1024*1024):.1f} MB")
    speed_text.text(f"⚡ Average Speed: {(total_size / (1024*1024)) / elapsed_time:.1f} MB/s")
    percent_text.text("📈 Progress: 100%")
    eta_text.text("⏱️ Upload Complete!")
    
    # Clear progress after showing completion
    time.sleep(2)
    progress_container.empty()
    
    return file_bytes, get_file_hash(file_bytes)


def main():
    st.set_page_config(
        page_title="Audio/Video Transcription & Translation",
        page_icon="🎵",
        layout="wide",
    )

    st.title("🎵 Audio/Video Transcription & Translation")
    st.markdown(
        "Upload audio or video files for real-time transcription and translation"
    )

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")

        transcription_language = st.selectbox(
            "Transcription Language (optional)",
            options=[None, "en", "ar", "fr"],
            format_func=lambda x: "Auto-detect" if x is None else x,
            help="Leave as 'Auto-detect' for automatic language detection",
        )

        enable_translation = st.checkbox("Enable Translation", value=True)

        if enable_translation:
            target_language = st.selectbox(
                "Target Language",
                options=["English", "French", "German", "Spanish"],
                index=0,
            )

    st.header("📁 File Upload")

    uploaded_file = st.file_uploader(
        "Choose an audio or video file",
        type=[
            "mp3",
            "wav",
            "m4a",
            "aac",
            "ogg",
            "flac",
            "mp4",
            "avi",
            "mov",
            "mkv",
            "wmv",
            "webm",
        ],
        help="Supported formats: Audio (MP3, WAV, M4A, AAC, OGG, FLAC) and Video (MP4, AVI, MOV, MKV, WMV, WEBM)",
    )

    if uploaded_file:
        # Upload file with progress tracking
        with st.spinner("📤 Preparing upload..."):
            file_bytes, file_hash = upload_file_with_progress(uploaded_file)
        
        if file_bytes is None:
            st.error("❌ Failed to upload file")
            return

        # Reset processing state when new file is uploaded
        if st.session_state.current_file_hash != file_hash:
            st.session_state.conversion_complete = False
            st.session_state.processed_audio_path = None
            st.session_state.current_file_hash = file_hash
            st.session_state.transcription_complete = False
            st.session_state.transcription_text = ""
            st.session_state.translation_complete = False
            st.session_state.translation_text = ""

        st.success(f"✅ File uploaded: {uploaded_file.name}")

        # Get cached file info
        with st.spinner("📊 Analyzing file..."):
            file_info = get_file_info(file_bytes, uploaded_file.name, file_hash)
            st.session_state.uploaded_file_info = file_info

        # Display file information
        display_file_info(file_info)

        # Conversion section
        st.header("🔄 File Conversion")

        if file_info["is_video"] and not st.session_state.conversion_complete:
            st.info(
                "🎬 Video file detected - conversion to audio required for transcription"
            )

            if st.button("🔄 Convert Video to Audio", type="primary"):
                convert_video_file(file_info)

        elif file_info["is_audio"] and not st.session_state.conversion_complete:
            st.info("🎵 Audio file detected - preparing for transcription")

            if st.button("📋 Prepare Audio File", type="primary"):
                prepare_audio_file(file_info)

        elif st.session_state.conversion_complete:
            st.success("✅ File conversion completed!")

            # Show audio file information
            if hasattr(st.session_state, "audio_file_info"):
                display_audio_info()

            # Processing section
            st.header("⚙️ Transcription")

            transcription_client = get_openai_client("2025-03-01-preview")
            if transcription_client:
                if not st.session_state.transcription_complete:
                    if st.button("🚀 Start Transcription", type="primary"):
                        start_transcription(
                            transcription_language, transcription_client
                        )
                else:
                    st.success("✅ Transcription completed!")

                    # Show transcription result
                    st.text_area(
                        "Transcription Result",
                        value=st.session_state.transcription_text,
                        height=200,
                        key="final_transcription",
                    )

                    # Translation section
                    st.header("🌐 Translation (Optional)")

                    translation_client = get_openai_client("2024-12-01-preview")

                    if enable_translation:
                        if not st.session_state.translation_complete:
                            if st.button("🌐 Start Translation", type="secondary"):
                                start_translation(target_language, translation_client)
                        else:
                            st.success("✅ Translation completed!")
                            st.text_area(
                                f"Translation Result ({target_language})",
                                value=st.session_state.translation_text,
                                height=200,
                                key="final_translation",
                            )
                    else:
                        st.info(
                            "💡 Enable translation in the sidebar to translate the transcription"
                        )

                    # Download section
                    st.header("💾 Download Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.download_button(
                            label="📄 Download Transcription",
                            data=st.session_state.transcription_text,
                            file_name=f"transcription_{st.session_state.uploaded_file_info['name']}.txt",
                            mime="text/plain",
                        )

                    if st.session_state.translation_complete:
                        with col2:
                            st.download_button(
                                label="🌐 Download Translation",
                                data=st.session_state.translation_text,
                                file_name=f"translation_{st.session_state.uploaded_file_info['name']}.txt",
                                mime="text/plain",
                            )
            else:
                st.warning("⚠️ Please enter your OpenAI API key in the sidebar")

    elif not uploaded_file:
        st.info("👆 Please upload a file to begin")


def display_file_info(file_info):
    """Display file information after upload"""
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**File size:** {file_info['size_mb']:.2f} MB")
        st.write(f"**Format:** {file_info['format'].upper()}")

    if file_info["is_video"]:
        with col2:
            if "duration" in file_info:
                duration_mins = file_info["duration"] / 60
                st.write(f"**Duration:** {duration_mins:.1f} minutes")
            if "resolution" in file_info:
                st.write(f"**Resolution:** {file_info['resolution']}")
            if "fps" in file_info:
                st.write(f"**FPS:** {file_info['fps']:.1f}")
            if "has_audio" in file_info:
                audio_status = "✅ Yes" if file_info["has_audio"] else "❌ No"
                st.write(f"**Has Audio:** {audio_status}")

                if not file_info["has_audio"]:
                    st.error("⚠️ This video file has no audio track!")


def convert_video_file(file_info):
    """Convert video file to audio with progress tracking"""
    if not file_info.get("has_audio", True):
        st.error("❌ Cannot process video without audio track!")
        return

    # Show estimated time
    if "duration" in file_info:
        estimated_time = file_info["duration"] / 60 * 0.1
        st.info(f"⏱️ Estimated conversion time: ~{estimated_time:.1f} minutes")

    # Create progress container
    progress_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("🎬 Starting conversion...")
            progress_bar.progress(20)

            # Use cached conversion function
            conversion_result = convert_video_to_audio_cached(
                file_info["hash"], file_info["temp_path"]
            )

            progress_bar.progress(80)
            status_text.text("🧹 Finalizing...")

            if conversion_result["success"]:
                # Store processed audio path
                st.session_state.processed_audio_path = conversion_result["audio_path"]
                st.session_state.conversion_complete = True

                # Store audio file information
                audio_info = {
                    "name": f"{Path(file_info['name']).stem}.wav",
                    "format": ".wav",
                    "size_mb": conversion_result["size_mb"],
                    "temp_path": conversion_result["audio_path"],
                    "is_audio": True,
                }
                st.session_state.audio_file_info = audio_info

                progress_bar.progress(100)
                status_text.text("✅ Audio extraction completed!")

                st.success(conversion_result["message"])

                # Clear progress after a moment
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()

                # Rerun to update UI
                st.rerun()
            else:
                st.error(conversion_result["message"])
                progress_bar.empty()
                status_text.empty()

        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            progress_bar.empty()
            status_text.empty()


def prepare_audio_file(file_info):
    """Prepare audio file for transcription"""
    with st.spinner("📋 Preparing audio file..."):
        # For audio files, we just use the uploaded file directly
        st.session_state.processed_audio_path = file_info["temp_path"]
        st.session_state.conversion_complete = True

        # Store audio file information (same as original for audio files)
        st.session_state.audio_file_info = file_info.copy()

        time.sleep(0.5)  # Brief pause for user feedback
        st.success("✅ Audio file prepared for transcription!")

        # Rerun to update UI
        st.rerun()


def display_audio_info():
    """Display audio file information after conversion"""
    st.subheader("🎵 Audio File Information")
    audio_info = st.session_state.audio_file_info

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Audio file:** {audio_info['name']}")
        st.write(f"**Format:** {audio_info['format'].upper()}")
        st.write(f"**Size:** {audio_info['size_mb']:.2f} MB")

    with col2:
        if "duration" in audio_info:
            duration_mins = audio_info["duration"] / 60
            st.write(f"**Duration:** {duration_mins:.1f} minutes")
        st.write(f"**Ready for transcription:** ✅")


def start_transcription(transcription_language, client):
    """Start the transcription process with real streaming"""
    if not st.session_state.processed_audio_path:
        st.error("❌ No audio file available for transcription")
        return

    # Create container for real-time updates
    transcription_container = st.container()

    with transcription_container:
        st.subheader("📝 Live Transcription")
        transcription_placeholder = st.empty()

    try:
        # Start real streaming transcription
        st.info("🎙️ Starting live transcription...")

        full_transcription = ""

        # Real streaming from OpenAI API
        for chunk in stream_transcription_real(
            client, st.session_state.processed_audio_path, transcription_language
        ):
            if chunk and not chunk.startswith("Error:"):
                full_transcription += chunk

                # Update the display in real-time
                transcription_placeholder.text_area(
                    "Live Transcription Stream",
                    value=full_transcription,
                    height=200,
                    key=f"live_transcription_{len(full_transcription)}",
                )

                # Small delay to make streaming visible
                time.sleep(0.01)
            elif chunk and chunk.startswith("Error:"):
                st.error(chunk)
                return

        # Store transcription results
        st.session_state.transcription_text = full_transcription
        st.session_state.transcription_complete = True

        st.success("✅ Transcription completed!")

        # Rerun to show the next steps
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error during transcription: {str(e)}")


def start_translation(target_language, client):
    """Start the translation process"""
    if not st.session_state.transcription_text:
        st.error("❌ No transcription available for translation")
        return

    try:
        st.info(f"🌐 Starting translation to {target_language}...")

        # Get cached translation
        text_hash = hashlib.md5(
            st.session_state.transcription_text.encode()
        ).hexdigest()
        translation_result = translate_text_cached(
            text_hash, st.session_state.transcription_text, client, target_language
        )

        if translation_result["success"]:
            # Store translation results
            st.session_state.translation_text = translation_result["text"]
            st.session_state.translation_complete = True

            st.success("✅ Translation completed!")

            # Rerun to show the translation result
            st.rerun()
        else:
            st.error(f"Translation error: {translation_result['error']}")

    except Exception as e:
        st.error(f"❌ Error during translation: {str(e)}")


if __name__ == "__main__":
    main()
