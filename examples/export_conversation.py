#!/usr/bin/env python3
"""
Export Conversation Example - Nexus-LLM
=========================================
Demonstrates how to export chat history in various formats
for backup, analysis, or sharing.
"""

import json
from datetime import datetime
from nexus_llm import InferenceEngine, Conversation
from nexus_llm.export import ConversationExporter, ExportFormat


def main():
    engine = InferenceEngine(model_name="nexus-7b-chat", device="auto")

    # Build a sample conversation
    conversation = Conversation(
        system_prompt="You are a helpful assistant specialized in Python programming.",
        metadata={
            "user_id": "user_123",
            "session_id": "session_abc",
            "created_at": datetime.now().isoformat(),
        }
    )

    conversation.add_user_message("How do I read a CSV file in Python?")
    response = engine.chat(conversation)
    conversation.add_assistant_message(response.text)

    conversation.add_user_message("What about using pandas?")
    response = engine.chat(conversation)
    conversation.add_assistant_message(response.text)

    conversation.add_user_message("Can you show me both approaches side by side?")
    response = engine.chat(conversation)
    conversation.add_assistant_message(response.text)

    # --- Export in different formats ---

    exporter = ConversationExporter()

    # 1. Export as JSON
    json_path = exporter.export(
        conversation,
        format=ExportFormat.JSON,
        output_path="./exports/conversation.json",
        include_metadata=True,
        include_token_counts=True,
    )
    print(f"Exported JSON: {json_path}")

    # 2. Export as Markdown
    md_path = exporter.export(
        conversation,
        format=ExportFormat.MARKDOWN,
        output_path="./exports/conversation.md",
        include_metadata=True,
    )
    print(f"Exported Markdown: {md_path}")

    # 3. Export as HTML
    html_path = exporter.export(
        conversation,
        format=ExportFormat.HTML,
        output_path="./exports/conversation.html",
        style="modern",            # Options: modern, classic, minimal
        include_metadata=True,
    )
    print(f"Exported HTML: {html_path}")

    # 4. Export as plain text
    txt_path = exporter.export(
        conversation,
        format=ExportFormat.TEXT,
        output_path="./exports/conversation.txt",
    )
    print(f"Exported Text: {txt_path}")

    # 5. Export for fine-tuning (JSONL format)
    jsonl_path = exporter.export(
        conversation,
        format=ExportFormat.JSONL_FINETUNE,
        output_path="./exports/fine_tune_data.jsonl",
        include_system_prompt=True,
    )
    print(f"Exported Fine-tune JSONL: {jsonl_path}")

    # 6. Export with custom template
    custom_template = """
========================================
Conversation: {metadata[session_id]}
Date: {metadata[created_at]}
========================================

{for message in messages}
[{message.role.upper()}] ({message.timestamp}):
{message.content}

{endfor}

---
Total tokens: {total_tokens}
"""

    custom_path = exporter.export_with_template(
        conversation,
        template=custom_template,
        output_path="./exports/conversation_custom.txt",
    )
    print(f"Exported Custom: {custom_path}")

    # --- Batch export all conversations ---
    conversations = Conversation.load_directory("./conversations/")
    batch_path = exporter.export_batch(
        conversations,
        format=ExportFormat.JSONL_FINETUNE,
        output_path="./exports/batch_finetune_data.jsonl",
    )
    print(f"\nBatch exported {len(conversations)} conversations to {batch_path}")


if __name__ == "__main__":
    main()
