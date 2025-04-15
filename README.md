# Chatbot 

This is a simple chatbot powered by Meta's [LLaMA-3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model. It runs directly from your terminal.

---

## Requirements

- Python **3.12**
- Create a virtual environment (optional but recommended):
```sh
python -m venv venv
source venv/bin/activate  
```

- Install dependencies:
```sh
pip install -r requirements.txt
```

- Hugging Face access to LLaMA-3: [Apply here](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
```sh
export HF_ACCESS_TOKEN='your_token'
```


## Usage

Run the chatbot:
```sh
python llama_chatbot.py
```

Example session:
LLaMA Chatbot is ready! Type your question below.
Type "exit" or "quit" to exit.

User: What is the capital of France
Bot: system

You are a helpful chatbot.
user

What is the capital of France
assistant

Bonjour! The capital of France is Paris!
