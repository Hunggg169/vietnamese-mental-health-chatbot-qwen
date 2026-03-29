# Vietnamese Mental Health Chatbot (LLM Fine-tuning)

## Overview
This project focuses on fine-tuning a Large Language Model (LLM) to build a Vietnamese mental health chatbot.  
The system generates supportive, context-aware responses using domain-specific data and can be integrated into real-time web applications.

---

## Model

- **Base Model:** Falcon  
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)  
- **Framework:** PyTorch, Hugging Face Transformers  

**Pretrained Model:**  
https://huggingface.co/HungHz/qwen2.5-1.5b-lora-fast

---

## Features

- Vietnamese conversational chatbot  
- Domain-specific responses for mental health support  
- Lightweight fine-tuned model using LoRA  
- Real-time interaction via web integration  

---

## Training Pipeline

1. Data collection and preprocessing  
2. Instruction-format dataset preparation  
3. Fine-tuning using LoRA  
4. Merging LoRA weights into base model  
5. Evaluation and testing  

---

## Tech Stack

- **Language:** Python  
- **Frameworks:** PyTorch, Transformers  
- **Techniques:** LoRA fine-tuning  
- **Deployment:** Flask / Node.js  
- **Database:** MongoDB  
