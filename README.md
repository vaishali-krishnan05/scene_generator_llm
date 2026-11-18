# Scene Generator LLM

An AI-based scene generation system that converts story ideas into
visual comic scenes using Large Language Models and image generation.

## Overview

The project takes a user's story premise and generates a structured
comic script containing characters, scenes, dialogue, and visual style.

The system supports local LLM backends such as LM Studio and Ollama.

## Main Features

- Story generation using an LLM
- Structured JSON output for comic scripts
- Input validation and content filtering
- RAG-based retrieval of existing comic stories
- FAISS similarity search
- Text embeddings using Sentence Transformers
- Image generation pipeline
- Streamlit-based user interface

## Project Structure

- `generate_scene.py` - Generates and validates comic story scripts.
- `RAG.py` - Handles comic retrieval using embeddings and FAISS.
- `streamlit_app.py` - Provides the Streamlit user interface.
- `Comic_project_commented.py` - Comic generation implementation.
- `requirements.txt` - Required Python packages.

## Technologies

- Python
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- FAISS
- Sentence Transformers
- Stable Diffusion / SDXL
- Streamlit
- PyTorch
- Ollama
- LM Studio

## How It Works

1. The user enters a story premise.
2. The input is validated to ensure it is suitable for story generation.
3. An LLM generates a structured comic script.
4. The generated content is validated as JSON.
5. RAG can retrieve relevant existing comic stories.
6. The scene information can then be used by the image-generation pipeline.
7. The generated result is presented through the Streamlit interface.

## Installation

```bash
git clone https://github.com/vaishali-krishnan05/scene_generator_llm.git
cd scene_generator_llm
pip install -r requirements.txt