# Voice Agent - Local LLM & Speech Processing

A comprehensive Python application for testing and experimenting with Large Language Models (LLMs), Speech-to-Text (STT), and Text-to-Speech (TTS) models locally.

## Features

- **Local LLM Integration**: Test various open-source LLM models (Llama, etc.) using GGUF format
- **Speech-to-Text**: Real-time audio transcription using OpenAI's Whisper
- **Text-to-Speech**: High-quality voice synthesis with multiple voices
- **Noise Suppression**: Clean audio input using DSP and WebRTC audio processing
- **Vector Database**: Semantic search and storage using Qdrant
- **Web GUI**: User-friendly interface for database management
- **Real-time Processing**: Live microphone input with minimal latency
- **Modular Architecture**: Easy to extend and customize

## Project Structure

```
├── llm_engine.py              # LLM inference engine
├── controller.py              # Main application controller
├── db_gui.py                  # Database GUI interface
├── vector_db.py               # Vector database operations
├── vector_embadings.py        # Embedding generation
├── dsp.py                     # Digital signal processing
├── mic_noise_suppressor.py    # Audio noise suppression
├── HuginfaceDown.py           # Hugging Face model downloader
├── start_DB_container.py      # Qdrant database container setup
├── speexdsp-python/           # Speex DSP library (noise cancellation)
├── version1.0/                # Legacy implementations
└── qdrant_storage/            # Vector database storage
```

## Requirements

### System Requirements
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for LLM inference)

### Dependencies
- OpenAI Whisper
- LLaMA-CPP-Python
- Qdrant (vector database)
- PyTorch
- WebRTC Audio Processing
- Pydantic
- WebSockets

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/RoshanAdi/Voice_agent.git
cd Voice_agent
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Models
Models need to be downloaded separately and placed in the project directory:

#### LLM Model (Llama)
```bash
# Download Llama-7B GGUF model
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

#### TTS Model
```bash
# Download Piper TTS model (already configured for en_US-amy-medium)
python HuginfaceDown.py
```

## Usage

### 1. Start Qdrant Database Container
```bash
python start_DB_container.py
```

### 2. Run the Application
```bash
python main.py
```

### 3. Web GUI (Database Manager)
```bash
python db_gui.py
```

### Real-Time Transcription
```bash
python version1.0/realtimeTranscribe.py
```

### Test Individual Models

#### Test LLM
```python
from llm_engine import LLMEngine

engine = LLMEngine(model_path="path/to/model.gguf")
response = engine.generate("What is AI?", max_tokens=100)
print(response)
```

#### Test Speech-to-Text
```python
from version1.0.NoiseFreeWhisper import WhisperTranscriber

transcriber = WhisperTranscriber()
text = transcriber.transcribe("audio.wav")
print(text)
```

#### Test Text-to-Speech
```python
from dsp import TTSEngine

tts = TTSEngine(voice="en_US-amy-medium")
tts.synthesize("Hello, World!", output_file="output.wav")
```

## Configuration

### Vector Database
- **Type**: Qdrant (Local deployment)
- **Collection**: `callcenter_bge`
- **Storage**: `./qdrant_storage/`

### Supported Models

#### LLM Models
- Llama 2 (7B, 13B, 70B)
- Mistral
- Any GGUF format compatible model

#### STT Models
- OpenAI Whisper (tiny, base, small, medium, large)

#### TTS Models
- Piper TTS (multiple voices and languages)
- Supported voices: en_US-amy-medium (configurable)

## Key Components

### llm_engine.py
Handles LLM inference with support for prompt templates, context management, and streaming responses.

### mic_noise_suppressor.py
Real-time audio noise suppression using WebRTC and Speex DSP algorithms for clean microphone input.

### vector_db.py
Interface for storing and retrieving embeddings using Qdrant, supporting semantic search and similarity queries.

### controller.py
Main orchestration layer managing the interaction between LLM, STT, TTS, and vector database components.

## Development

### Running Tests
```bash
# Test specific components
python -m pytest tests/
```

### Adding New Models
1. Place GGUF model files in project root or subdirectories
2. Update `llm_engine.py` with new model paths
3. Configure model-specific parameters in config files

### Extending Functionality
- Add new TTS voices in `dsp.py`
- Implement custom prompt templates in `llm_engine.py`
- Integrate additional vector stores in `vector_db.py`

## Troubleshooting

### Out of Memory Errors
- Reduce LLM model size (use smaller quantized versions)
- Lower batch size in configuration
- Use GPU acceleration (CUDA)

### Audio Quality Issues
- Check microphone permissions
- Adjust noise suppression sensitivity in `mic_noise_suppressor.py`
- Use higher quality audio input (16kHz or higher)

### Slow Inference
- Use smaller model quantization (Q4 vs Q8)
- Enable GPU acceleration if available
- Reduce context window size

## Performance Notes

- **Whisper Transcription**: ~5-10s for 1min audio (depends on model size)
- **Llama 7B Inference**: ~5-20 tokens/second (CPU), ~50+ tokens/second (GPU)
- **TTS Synthesis**: Real-time capable with Piper

## Architecture

```
User Input (Audio/Text)
    ↓
[Noise Suppression] ← (mic_noise_suppressor.py)
    ↓
[Speech-to-Text] ← (Whisper/NoiseFreeWhisper.py)
    ↓
[Vector Embedding] ← (vector_embadings.py)
    ↓
[Semantic Search] ← (vector_db.py/Qdrant)
    ↓
[LLM Processing] ← (llm_engine.py)
    ↓
[Text-to-Speech] ← (dsp.py)
    ↓
Audio Output
```

## Version History

- **v1.5**: Current stable release with improved noise suppression
- **v1.4**: Added WebRTC audio processing
- **v1.3**: Integrated Qdrant vector database
- **v1.2**: Multi-voice TTS support
- **v1.1**: Real-time transcription
- **v1.0**: Initial release

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Requirements File Example

See `requirements.txt` for complete dependency list. Key packages:
- openai-whisper
- llama-cpp-python
- qdrant-client
- torch
- pydantic
- websockets
- webrtcvad
- numpy
- scipy

## License

MIT License - See LICENSE file for details

## Support & Resources

- **OpenAI Whisper**: https://github.com/openai/whisper
- **Llama.cpp**: https://github.com/ggerganov/llama.cpp
- **Qdrant**: https://qdrant.tech/
- **Piper TTS**: https://github.com/rhasspy/piper

## Authors

**Roshan Mahesh Adi**
- Email: roshanmahesh1@gmail.com
- GitHub: [@RoshanAdi](https://github.com/RoshanAdi)

## Acknowledgments

- OpenAI for Whisper
- Meta for Llama
- Hugging Face community for model distributions
- Qdrant for vector database
- Rhasspy project for Piper TTS

---

**Last Updated**: October 2026

For issues, feature requests, or questions, please open an issue on GitHub or contact the author.
