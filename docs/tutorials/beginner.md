# Beginner Tutorial

Welcome to Nexus-LLM! This tutorial will walk you through installation, your first chat session, and basic commands. No prior experience with LLMs required.

---

## What You'll Learn

- How to install Nexus-LLM
- How to download and use your first model
- How to have a conversation with an AI
- How to use basic chat commands
- How to start the API server

**Time to complete:** ~15 minutes

---

## Part 1: Installation

### Step 1: Check Your Python Version

Nexus-LLM requires Python 3.10 or later. Check your version:

```bash
python3 --version
```

You should see something like `Python 3.11.5`. If your version is older than 3.10, upgrade Python first.

### Step 2: Clone the Repository

```bash
git clone https://github.com/brutal-45/Nexus-LLM.git
cd nexus-llm
```

### Step 3: Run the Installer

```bash
chmod +x scripts/*.sh
./scripts/install.sh
```

You'll see output like:

```
[INFO]  Detecting Python installation...
[INFO]  Found Python 3.11.5 at /usr/bin/python3
[OK]    Python version 3.11.5 meets requirements (>= 3.10)
[INFO]  Creating virtual environment in /home/user/nexus-llm/.venv
[OK]    Virtual environment created
[INFO]  Upgrading pip, setuptools, and wheel...
[OK]    Core packages upgraded
[INFO]  Installing dependencies from requirements.txt...
[OK]    Core dependencies installed
[OK]    Nexus-LLM installed successfully!
```

If anything fails, the script will tell you what went wrong and how to fix it.

### Step 4: Verify the Installation

```bash
# Activate the virtual environment
source .venv/bin/activate

# Check that the CLI works
nexus --version
```

---

## Part 2: Download a Model

Before you can chat, you need a model. For beginners, we recommend **Phi-3 Mini** — it's small (7GB), fast, and runs on most GPUs.

### Get a HuggingFace Token

Some models (like Llama) require a free HuggingFace account:

1. Go to [huggingface.co](https://huggingface.co) and sign up
2. Go to **Settings → Access Tokens** → **New token**
3. Name it "nexus-llm" and give it **Read** permission
4. Copy the token

Add it to your `.env` file:

```bash
echo 'NEXUS_HF_TOKEN=hf_paste_your_token_here' >> .env
```

### Download the Model

```bash
./scripts/download_models.sh phi-3-mini
```

This downloads ~7GB. You'll see:

```
[INFO]  Downloading model: microsoft/Phi-3-mini-4k-instruct
Downloading: 100%|████████████████| 7.1GB/7.1GB [02:30<00:00, 47.3MB/s]
[OK]    Successfully downloaded: microsoft/Phi-3-mini-4k-instruct
```

> **No GPU?** You can still use Nexus-LLM on CPU — it'll just be slower. Or try an even smaller model.

---

## Part 3: Your First Chat

### Start the Chat

```bash
./scripts/run.sh --mode chat
```

The first time you run this, the model loads into memory (takes 10–30 seconds). Then you'll see:

```
╔══════════════════════════════════════════╗
║          Nexus-LLM Chat Mode             ║
╠══════════════════════════════════════════╣
║  Model: meta-llama/Llama-3.1-8B-Instruct ║
║  Device: cuda:0                           ║
║  Type /help for commands, /quit to exit   ║
╚══════════════════════════════════════════╝
```

### Say Hello

Type a message and press Enter:

```
You> Hello! What can you help me with?

Assistant> Hello! I can help you with a wide range of tasks, including:

1. **Answering questions** - From general knowledge to specific topics
2. **Writing assistance** - Drafting emails, essays, creative writing
3. **Code help** - Writing, debugging, and explaining code
4. **Analysis** - Breaking down complex topics
5. **Brainstorming** - Generating ideas and solutions

What would you like to explore?
```

### Try Different Types of Questions

```
You> Explain how rainbows form, like I'm 10 years old.

Assistant> Imagine sunlight is like a secret code made of all the colors
mixed together — red, orange, yellow, green, blue, and purple. When white
sunlight hits a raindrop floating in the air after a rain, something magical
happens...

The raindrop acts like a tiny prism (a special glass shape). When light
goes in, the raindrop bends each color differently because each color
travels at a slightly different speed. This separates them out, like
sorting M&Ms by color!

Then the colors bounce off the back of the raindrop and come back out
toward you. Since there are millions of raindrops in the sky, each one
sends you a different color at a slightly different angle. Red comes from
higher raindrops, and purple from lower ones — that's why you see them
in that order as a big arc across the sky!

So next time you see a rainbow, you're actually seeing sunlight that
got sorted into its individual colors by tiny floating water prisms!
```

### Use a System Prompt

Set a custom system prompt to change the AI's personality:

```
You> /system You are a pirate captain. Always speak in pirate dialect.

System prompt updated.

You> Tell me about the weather today.

Assistant> Arrr, me heartie! The skies be clear as a mermaid's tear today,
with a fine easterly wind blowin' through me sails! 'Tis a perfect day for
sailin' the seven seas, says I. The sun be shinin' bright and the waves
be calm — a fine day indeed for plunderin'... er, I mean, explorin'!

Yarr, check yer local weather map for the details of yer particular port,
as I be sailin' the digital seas and cannae see yer actual sky! 🏴‍☠️
```

---

## Part 4: Basic Commands

Here are the most useful chat commands:

### View and Change Settings

```
You> /info
Model: microsoft/Phi-3-mini-4k-instruct
Device: cuda:0
Temperature: 0.7
Top-p: 0.9
Max tokens: 2048

You> /temp 0.3
Temperature set to 0.3

You> /maxtokens 500
Max tokens set to 500
```

### Save and Load Conversations

```
You> /save my_chat
Conversation saved to: ./data/conversations/my_chat.json

You> /load my_chat
Conversation loaded from: ./data/conversations/my_chat.json
```

### Clear and Start Fresh

```
You> /clear
Conversation history cleared.

You> /system You are a helpful math tutor.
System prompt updated.
```

### Switch Models

```
You> /model mistralai/Mistral-7B-Instruct-v0.3
Loading model... (this may take a moment)
Model switched to: mistralai/Mistral-7B-Instruct-v0.3
```

---

## Part 5: Start the API Server

You can also run Nexus-LLM as an API server, which lets you call it from any programming language.

### Start the Server

```bash
./scripts/run.sh --mode server
```

```
[INFO]  Starting API server on 127.0.0.1:8000
[INFO]  Nexus-LLM API v1.2.0 running
```

### Test with curl

```bash
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/Phi-3-mini-4k-instruct",
    "messages": [
      {"role": "user", "content": "Hello from the API!"}
    ]
  }'
```

### Test with Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/chat/completions",
    json={
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"}
        ]
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

---

## What's Next?

Congratulations! You've completed the beginner tutorial. You now know how to:

- ✅ Install and set up Nexus-LLM
- ✅ Download models
- ✅ Chat with an AI
- ✅ Use basic commands
- ✅ Run the API server

Ready for more? Continue with:

- **[Intermediate Tutorial](./intermediate.md)** — Learn fine-tuning, custom prompts, and RAG
- **[Quick Start Guide](../guides/quickstart.md)** — Comprehensive setup reference
- **[Configuration Guide](../guides/configuration.md)** — Customize all settings
