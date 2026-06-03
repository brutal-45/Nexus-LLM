#!/usr/bin/env python3
"""
Multimodal Pipeline Example - Nexus-LLM
=========================================
Demonstrates how to use multimodal capabilities for processing
text, images, and audio inputs.
"""

from nexus_llm import InferenceEngine, Conversation
from nexus_llm.multimodal import (
    MultimodalPipeline,
    ImageInput,
    AudioInput,
    VisionEncoder,
    AudioEncoder,
)


def main():
    # --- Initialize the multimodal pipeline ---
    pipeline = MultimodalPipeline(
        text_model="nexus-7b-chat",
        vision_encoder="nexus-vision-large",
        audio_encoder="nexus-audio-base",
        device="auto",
        fusion_strategy="cross_attention",  # Options: cross_attention, concatenation, gating
    )

    # --- Image understanding ---
    print("=" * 60)
    print("Image Understanding")
    print("=" * 60)

    # From a file path
    image = ImageInput.from_file("./test_images/chart.png")
    result = pipeline.query(
        text="Describe what you see in this image in detail.",
        image=image,
    )
    print(f"Image Description: {result.text}\n")

    # From a URL
    image_url = ImageInput.from_url("https://example.com/photo.jpg")
    result = pipeline.query(
        text="What objects are visible in this photo?",
        image=image_url,
    )
    print(f"Object Detection: {result.text}\n")

    # Multiple images
    images = [
        ImageInput.from_file("./test_images/before.png"),
        ImageInput.from_file("./test_images/after.png"),
    ]
    result = pipeline.query(
        text="Compare these two images and describe the differences.",
        images=images,
    )
    print(f"Image Comparison: {result.text}\n")

    # --- Audio understanding ---
    print("=" * 60)
    print("Audio Understanding")
    print("=" * 60)

    # Transcribe audio
    audio = AudioInput.from_file("./test_audio/meeting.wav")
    result = pipeline.query(
        text="Transcribe this audio and provide a summary.",
        audio=audio,
    )
    print(f"Audio Summary: {result.text}\n")

    # Audio with specific instructions
    result = pipeline.query(
        text="What language is being spoken in this audio? "
             "Also identify the speaker's emotional tone.",
        audio=AudioInput.from_file("./test_audio/speech.mp3"),
    )
    print(f"Language & Emotion: {result.text}\n")

    # --- Combined multimodal input ---
    print("=" * 60)
    print("Combined Multimodal Input")
    print("=" * 60)

    # Image + Audio + Text
    result = pipeline.query(
        text="This image and audio were captured at the same event. "
             "Describe how the visual content relates to what is being said.",
        image=ImageInput.from_file("./test_images/event_photo.jpg"),
        audio=AudioInput.from_file("./test_audio/event_speech.wav"),
    )
    print(f"Multimodal Analysis: {result.text}\n")

    # --- Conversational multimodal ---
    print("=" * 60)
    print("Conversational Multimodal")
    print("=" * 60)

    conversation = Conversation(
        system_prompt="You are a helpful assistant that can understand images and audio."
    )

    # First turn: image
    conversation.add_user_message(
        "What does this chart show?",
        image=ImageInput.from_file("./test_images/sales_chart.png"),
    )
    response = pipeline.chat(conversation)
    print(f"Turn 1 - Chart analysis: {response.text}\n")

    # Second turn: follow-up (no new image needed)
    conversation.add_user_message("What were the peak sales months?")
    response = pipeline.chat(conversation)
    print(f"Turn 2 - Follow-up: {response.text}\n")

    # --- Batch processing ---
    print("=" * 60)
    print("Batch Image Processing")
    print("=" * 60)

    batch_images = [
        ImageInput.from_file(f"./test_images/img_{i}.jpg")
        for i in range(1, 6)
    ]

    batch_results = pipeline.batch_query(
        texts=["Describe this image."] * len(batch_images),
        images=batch_images,
        batch_size=2,
    )

    for i, result in enumerate(batch_results, 1):
        print(f"Image {i}: {result.text[:100]}...")


if __name__ == "__main__":
    main()
