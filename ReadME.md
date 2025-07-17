# 🎵 Meeting Transcription Agent

A powerful Streamlit application for transcribing and translating audio/video files using Azure OpenAI services. This application provides real-time transcription with optional translation capabilities for meeting recordings and other audio content.

## ✨ Features

- **Multi-format Support**: Audio (MP3, WAV, M4A, AAC, OGG, FLAC) and Video (MP4, AVI, MOV, MKV, WMV, WEBM)
- **Real-time Transcription**: Streaming transcription using Azure OpenAI's GPT-4o-transcribe model
- **Smart Chunking**: Automatically handles large files by splitting them into processable chunks
- **Multi-language Support**: Auto-detect or specify transcription language (English, Arabic, French)
- **Translation**: Contextual translation to multiple target languages
- **Progress Tracking**: Real-time progress indicators for all operations
- **State Management**: Maintains processing state across sessions
- **Download Results**: Export transcriptions and translations as text files

## 🛠️ Prerequisites

- **Python 3.11** (required for optimal compatibility)
- **Azure OpenAI Account** with API access
- **FFmpeg** (for video processing - automatically handled by moviepy)

## 📋 Setup Instructions

### 1. Clone or Download the Project

```bash
git clone https://github.com/mhsFlairs/meeting-transcriber
cd MeetingTrascriptionAgent
```

### 2. Create Python Virtual Environment

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory:

```bash
# Copy the example and edit with your credentials
cp .env.example .env
```

Edit the `.env` file with your Azure OpenAI credentials:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
```

**To get your Azure OpenAI credentials:**
1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your Azure OpenAI resource
3. Go to "Keys and Endpoint" section
4. Copy the endpoint URL and API key

### 5. Install Additional System Dependencies (if needed)

**For video processing (usually auto-installed):**
```bash
# FFmpeg is typically installed automatically with moviepy
# If you encounter issues, install manually:

# Windows (using chocolatey):
choco install ffmpeg

# macOS (using homebrew):
brew install ffmpeg

# Ubuntu/Debian:
sudo apt update
sudo apt install ffmpeg
```

## 🚀 Running the Application

### Start the Streamlit Application

```bash
# Ensure virtual environment is activated
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Run the application
streamlit run app.py
```

### Alternative Run Methods

**Using Python directly:**
```bash
python -m streamlit run app.py
```

**With custom port:**
```bash
streamlit run app.py --server.port 8502
```

**With custom host (for network access):**
```bash
streamlit run app.py --server.address 0.0.0.0
```

## 🎯 Usage Guide

### Basic Workflow

1. **Upload File**: Choose an audio or video file
2. **File Analysis**: View file information and metadata
3. **Conversion**: Convert video to audio (if needed)
4. **Transcription**: Real-time transcription with progress tracking
5. **Translation**: Optional translation to target language
6. **Download**: Export results as text files

### Supported File Formats

**Audio Files:**
- MP3, WAV, M4A, AAC, OGG, FLAC, WMA

**Video Files:**
- MP4, AVI, MOV, MKV, WMV, FLV, WEBM, M4V

### Configuration Options

**Sidebar Settings:**
- **Transcription Language**: Auto-detect, English, Arabic, French
- **Enable Translation**: Toggle translation feature
- **Target Language**: English, French, German, Spanish

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure virtual environment is activated and dependencies installed
.venv\Scripts\activate
pip install -r requirements.txt
```

**2. Azure OpenAI Connection Issues**
- Verify your `.env` file has correct credentials
- Check Azure OpenAI resource is active and has model deployments
- Ensure API key has proper permissions

**3. Video Processing Errors**
```bash
# Install FFmpeg manually if auto-installation fails
# Windows:
choco install ffmpeg
# macOS:
brew install ffmpeg
# Linux:
sudo apt install ffmpeg
```

**4. Large File Processing**
- Files over 25MB are automatically chunked
- Ensure sufficient disk space for temporary files
- Check logs in `transcription_app.log` for detailed error information

**5. Memory Issues**
```bash
# For large files, consider increasing system memory or
# processing smaller chunks by modifying chunk_size_mb in app.py
```

### Logging

Application logs are automatically saved to `transcription_app.log` with detailed information about:
- File processing steps
- Error messages and stack traces
- Performance metrics
- API call details

## 📋 Requirements File

The `requirements.txt` should contain:

```txt
streamlit>=1.28.0
openai>=1.0.0
moviepy>=1.0.3
python-dotenv>=1.0.0
pydub>=0.25.1
pathlib>=1.0.1
```

## 🔐 Security Notes

- **Never commit your `.env` file** to version control
- **Keep your Azure OpenAI API keys secure**
- **Regularly rotate your API keys**
- **Monitor your Azure OpenAI usage and costs**

## 📊 Performance Tips

- **File Size**: Optimal performance with files under 100MB
- **Formats**: MP3 and MP4 typically process fastest
- **Network**: Stable internet connection required for API calls
- **Hardware**: SSD storage recommended for temporary file processing

## 🆘 Getting Help

1. **Check Logs**: Review `transcription_app.log` for detailed error information
2. **Verify Setup**: Ensure all prerequisites and dependencies are installed
3. **Test Connection**: Verify Azure OpenAI credentials and connectivity
4. **File Format**: Confirm your file format is supported

## 📜 License

This project is provided as-is for educational and development purposes.

## 🔄 Updates

To update the application:

```bash
# Activate virtual environment
.venv\Scripts\activate

# Pull latest changes (if using git)
git pull

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart the application
streamlit run app.py
```