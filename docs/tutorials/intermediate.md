# Intermediate Tutorial

Take your Nexus-LLM skills further with fine-tuning, custom prompt engineering, and retrieval-augmented generation (RAG).

---

## Prerequisites

- Completed the [Beginner Tutorial](./beginner.md)
- Nexus-LLM installed and a model downloaded
- Basic familiarity with the command line and Python

**Time to complete:** ~45 minutes

---

## Part 1: Fine-Tuning a Model

Fine-tuning adapts a pre-trained model to your specific use case. We'll use LoRA — a fast, memory-efficient method.

### Step 1: Prepare Training Data

Create a JSONL file with your training examples. Each line should be a JSON object with a `messages` array:

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful cooking assistant."}, {"role": "user", "content": "How do I make pasta carbonara?"}, {"role": "assistant", "content": "Classic pasta carbonara is simple and delicious! Here's how:\n\nIngredients:\n- 400g spaghetti\n- 200g guanciale (or pancetta)\n- 4 egg yolks + 1 whole egg\n- 100g Pecorino Romano, grated\n- Black pepper\n\n1. Cook spaghetti in salted boiling water until al dente\n2. Cut guanciale into strips and fry until crispy\n3. Whisk eggs, yolks, and cheese together\n4. Drain pasta (save some pasta water!)\n5. Toss hot pasta with guanciale\n6. Remove from heat, add egg mixture, toss quickly\n7. Add pasta water if needed for creaminess\n8. Serve with extra cheese and black pepper"}]}
{"messages": [{"role": "system", "content": "You are a helpful cooking assistant."}, {"role": "user", "content": "What's the difference between baking soda and baking powder?"}, {"role": "assistant", "content": "Baking soda and baking powder are both leavening agents, but they work differently:\n\n**Baking soda** (sodium bicarbonate) needs an acid to activate. When combined with acidic ingredients like buttermilk, vinegar, or lemon juice, it produces carbon dioxide bubbles that make baked goods rise.\n\n**Baking powder** contains baking soda PLUS an acid (cream of tartar). It activates in two stages: once when mixed with liquid, and again when heated. Most baking powder sold today is \"double-acting.\"\n\n**Rule of thumb:**\n- If your recipe has an acidic ingredient → use baking soda\n- If no acidic ingredient → use baking powder\n- Never substitute them 1:1!\n\nSubstitution: 1 tsp baking powder = 1/4 tsp baking soda + 1/2 tsp cream of tartar"}]}
```

Save this as `data/cooking_assistant.jsonl`. For real training, aim for 200+ examples.

### Step 2: Split Your Data

```bash
nexus data split --input data/cooking_assistant.jsonl --test-ratio 0.1 --seed 42
```

This creates:
- `data/cooking_assistant_train.jsonl` (90%)
- `data/cooking_assistant_eval.jsonl` (10%)

### Step 3: Run LoRA Fine-Tuning

```bash
./scripts/train.sh \
  --model microsoft/Phi-3-mini-4k-instruct \
  --dataset data/cooking_assistant_train.jsonl \
  --method lora \
  --rank 8 \
  --alpha 16 \
  --epochs 3 \
  --batch-size 4 \
  --lr 2e-5 \
  --output-dir checkpoints/cooking_assistant
```

Training output:

```
╔══════════════════════════════════════════════════════╗
║            Nexus-LLM Training Pipeline               ║
╠══════════════════════════════════════════════════════╣
║  Model          : microsoft/Phi-3-mini-4k-instruct   ║
║  Dataset        : data/cooking_assistant_train.jsonl ║
║  Method         : lora                                ║
║  Epochs         : 3                                   ║
║  Batch size     : 4                                   ║
║  Grad accum     : 4                                   ║
║  Learning rate  : 2e-5                                ║
╚══════════════════════════════════════════════════════╝

Training: 100%|████████████████████| 150/150 [12:30<00:00, 0.20it/s]
Epoch 1/3 | Step 50/150 | Loss: 1.842 | LR: 1.9e-5
Epoch 2/3 | Step 100/150 | Loss: 0.921 | LR: 1.2e-5
Epoch 3/3 | Step 150/150 | Loss: 0.654 | LR: 0.0e+0

[OK]    Training completed successfully!
[OK]    Checkpoints saved to: checkpoints/cooking_assistant
```

### Step 4: Test Your Fine-Tuned Model

```bash
./scripts/run.sh --mode chat \
  --model microsoft/Phi-3-mini-4k-instruct \
  --adapter checkpoints/cooking_assistant
```

Now ask cooking questions and compare the responses to the base model.

---

## Part 2: Prompt Engineering

Crafting effective prompts dramatically improves response quality.

### System Prompts

The system prompt sets the AI's behavior for the entire conversation:

```
You> /system You are an expert Python developer. When answering questions:
1. Always provide working code examples
2. Explain your reasoning step by step
3. Mention edge cases and error handling
4. Follow PEP 8 style guidelines

System prompt updated.

You> How do I read a CSV file?

Assistant> Here's how to read a CSV file in Python, step by step:

**Using the `csv` module (built-in):**

```python
import csv
from pathlib import Path

def read_csv(filepath: str) -> list[dict]:
    """Read a CSV file and return a list of dictionaries.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        List of row dictionaries with header keys
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    with path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    if not rows:
        raise ValueError("CSV file is empty")
    
    return rows

# Usage
try:
    data = read_csv("sales_data.csv")
    for row in data:
        print(row)
except (FileNotFoundError, ValueError) as e:
    print(f"Error: {e}")
```

**Edge cases handled:**
- Missing file → `FileNotFoundError`
- Empty file → `ValueError`
- Encoding issues → explicit UTF-8 encoding
- Newline handling → `newline=''` per csv module docs
```

### Few-Shot Prompting

Include examples in your message to guide the model:

```
You> Convert these descriptions to SQL queries:

Description: Get all users who signed up last month
SQL: SELECT * FROM users WHERE created_at >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH);

Description: Count orders by status
SQL: SELECT status, COUNT(*) FROM orders GROUP BY status;

Description: Find the top 5 products by revenue
SQL:
```

### Chain-of-Thought Prompting

Ask the model to reason step by step:

```
You> A store has 15 apples. They sell 3 in the morning, then receive a
delivery of 8 more, and sell 5 in the afternoon. How many apples remain?
Think step by step.

Assistant> Let me work through this step by step:

1. Start with 15 apples
2. Sell 3 in the morning: 15 - 3 = 12 apples
3. Receive delivery of 8: 12 + 8 = 20 apples
4. Sell 5 in the afternoon: 20 - 5 = 15 apples

**Answer: 15 apples remain.**
```

### Creating Prompt Templates

Save reusable prompt templates:

```yaml
# config/prompts/code_review.yaml
name: "code_review"
description: "Review code and provide feedback"
template: |
  You are a senior software engineer performing a code review.
  Analyze the following code for:

  1. **Bugs and errors** - Logic errors, null references, off-by-one errors
  2. **Security issues** - SQL injection, XSS, hardcoded secrets
  3. **Performance** - Inefficient algorithms, unnecessary allocations
  4. **Style** - Naming, formatting, adherence to best practices
  5. **Maintainability** - Complexity, testability, documentation

  Code to review:
  ```
  {code}
  ```

  Provide specific, actionable feedback with line references.
```

Use the template:

```bash
nexus prompt code_review --var code="$(cat my_script.py)"
```

---

## Part 3: Retrieval-Augmented Generation (RAG)

RAG lets the AI answer questions using your own documents.

### Step 1: Enable RAG

Edit `config/user.yaml`:

```yaml
rag:
  enabled: true
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 512
  chunk_overlap: 64
  retrieval_top_k: 5
```

### Step 2: Index Your Documents

```bash
# Index a directory of documents
nexus rag index --dir ./documents/ --collection my_docs

# Index a single file
nexus rag index --file ./documents/product_manual.pdf --collection my_docs
```

### Step 3: Query with RAG

```bash
./scripts/run.sh --mode chat
```

```
You> /rag on
RAG enabled. Using collection: my_docs

You> What is our return policy?

Assistant> Based on the product documentation:

Our return policy allows customers to return items within 30 days of
purchase for a full refund. Items must be in their original packaging
and unused condition.

For electronics, the return window is 15 days. Software licenses are
non-refundable once activated.

To initiate a return, visit our returns portal at returns.example.com
or contact support.

[Source: product_manual.pdf, Page 23]
```

### Step 4: Manage Collections

```bash
# List all collections
nexus rag list-collections

# Check collection stats
nexus rag stats --collection my_docs

# Add more documents to an existing collection
nexus rag index --file ./documents/faq.md --collection my_docs

# Delete a collection
nexus rag delete-collection --name old_docs
```

---

## Part 4: Building a Simple Application

Let's build a Python script that uses the Nexus-LLM API to create a question-answering bot.

```python
# qa_bot.py
import requests
import sys

class QABot:
    """Simple question-answering bot using Nexus-LLM API."""

    def __init__(self, base_url="http://localhost:8000", api_key="nexus_your_api_key"):
        self.base_url = base_url
        self.api_key = api_key
        self.history = []

    def ask(self, question: str) -> str:
        """Ask a question and get an answer."""
        self.history.append({"role": "user", "content": question})

        response = requests.post(
            f"{self.base_url}/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "microsoft/Phi-3-mini-4k-instruct",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Be concise and accurate."},
                    *self.history[-10:],  # Keep last 10 messages for context
                ],
                "temperature": 0.7,
                "max_tokens": 500,
            },
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        self.history.append({"role": "assistant", "content": answer})
        return answer

    def clear_history(self):
        """Clear conversation history."""
        self.history = []


if __name__ == "__main__":
    bot = QABot()

    print("QA Bot - Type 'quit' to exit, 'clear' to reset\n")

    while True:
        question = input("You> ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if question.lower() == "clear":
            bot.clear_history()
            print("History cleared.\n")
            continue
        if not question:
            continue

        try:
            answer = bot.ask(question)
            print(f"\nBot> {answer}\n")
        except requests.exceptions.RequestException as e:
            print(f"\nError: {e}\n")
```

Run it:

```bash
python qa_bot.py
```

---

## What's Next?

You've completed the intermediate tutorial! You now know how to:

- ✅ Fine-tune a model with LoRA
- ✅ Craft effective prompts
- ✅ Set up RAG for document-based Q&A
- ✅ Build applications using the API

Continue with:

- **[Advanced Tutorial](./advanced.md)** — Distributed training, quantization, custom agents
- **[Fine-Tuning Guide](../guides/fine_tuning.md)** — Deep dive into LoRA/QLoRA
- **[RAG Guide](../guides/rag.md)** — Advanced RAG techniques
