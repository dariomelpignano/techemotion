---
title: Tech.Emotion Summit 2025 Virtual Assistant
emoji: 🤖
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 4.29.0
app_file: app.py
pinned: false
---

# Tech.Emotion Summit 2025 Virtual Assistant

A virtual assistant built for the Tech.Emotion Summit 2025, powered by OpenAI and LangChain.

## Setup

1. Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4
EMBED_MODEL=text-embedding-3-small
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
python app.py
```

## Deployment

This app can be deployed on Gradio Spaces by:
1. Creating a new Space on Hugging Face
2. Connecting your GitHub repository
3. Setting the environment variables in the Space settings 