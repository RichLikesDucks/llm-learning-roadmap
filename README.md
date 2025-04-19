# LLM Learning Roadmap — Project-Based Guide

This is a hands-on, project-based roadmap to deeply understand how Large Language Models (LLMs) work — from tokenization and Transformers to fine-tuning, RAG systems, and model serving.

## ⚡️ Phase 1: Foundations

### 1. Build a Transformer from Scratch
- Implement self-attention, positional encodings, masking, and FFN layers using NumPy.
- Reference: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), [Karpathy’s minGPT](https://github.com/karpathy/minGPT)

### 2. Byte Pair Encoding (BPE) Tokenizer
- Build a BPE tokenizer from scratch on a small dataset.
- Visualize merges, vocab growth, and tokenization outputs.

### 3. Character-Level Language Model
- Train a small RNN or Transformer on a corpus (e.g. poems, lyrics).
- Try different decoding strategies (greedy, sampling, top-k).

---

## 🚀 Phase 2: Apply and Extend

### 4. Fine-Tune GPT-2 on Custom Dataset
- Use Hugging Face `transformers` to fine-tune GPT-2 on domain-specific data.
- Use `Trainer`, `LoRA`, or `PEFT` techniques for efficient fine-tuning.

### 5. Local LLM Chatbot
- Load a quantized model (e.g. TinyLlama, Mistral) using `transformers` or `llama.cpp`.
- Build a chat interface with Gradio or Streamlit.
- Add chat memory + context window logic.

### 6. Visualize Attention Heads
- Hook into a transformer model and visualize attention scores.
- Compare how different heads and layers behave across inputs.

---

## 🧠 Phase 3: Systems & Scaling

### 7. Retrieval-Augmented Generation (RAG)
- Use FAISS/ChromaDB + sentence transformers to retrieve relevant chunks.
- Inject context into LLM prompts to enhance QA and summarization.

### 8. Train a Small LLM (125M Params)
- Use Hugging Face `transformers`, `accelerate`, `DeepSpeed` or `FSDP`.
- Train on a cleaned corpus like OpenWebText.
- Track training loss, validation perplexity, and sampling quality.

### 9. LLM Model Serving
- Deploy an LLM backend with FastAPI + Docker + GPU support.
- Add batching, token streaming, and prompt cache optimizations.

---

## 🧪 Bonus Projects

- **LLM Agents**: Implement a simple tool-using agent with a memory module.
- **Chain-of-Thought Prompting**: Evaluate multi-step reasoning performance.
- **In-Context Learning Experiments**: Compare zero-shot vs few-shot results.

---

## 🛠 Stack & Tools

- **Core Libraries**: PyTorch, Hugging Face Transformers, NumPy
- **Infra & Deployment**: Docker, FastAPI, Gradio, Streamlit
- **Optional**: DeepSpeed, FSDP, llama.cpp, LangChain, Weights & Biases

---

## ✅ Progress Tracker

| Project | Status | Notes |
|--------|--------|-------|
| Transformer from scratch | ☐ | |
| BPE Tokenizer | ☐ | |
| Char-level LM | ☐ | |
| Fine-tune GPT-2 | ☐ | |
| Chatbot UI | ☐ | |
| Attention Visualization | ☐ | |
| RAG System | ☐ | |
| Train 125M model | ☐ | |
| Model Serving | ☐ | |

---

## License

MIT
