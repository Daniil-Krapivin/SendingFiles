"""
title: Qwen Vision Preprocessor
author: user
author_url: https://github.com/open-webui
version: 0.1
"""

from pydantic import BaseModel, Field
from typing import Optional
import requests


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        vision_api_url: str = Field(
            default="http://vllm-qwen:8000/v1",
            description="Qwen vLLM server URL"
        )
        vision_model: str = Field(
            default="Qwen/Qwen2.5-VL-3B-Instruct",
            description="Vision model name"
        )
        extraction_prompt: str = Field(
            default="Extract all text from this image verbatim. If no text, briefly describe what you see.",
            description="Prompt sent to vision model"
        )
        pass

    class UserValves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    def _process_image(self, image_data: str) -> str:
        """Send image to Qwen and get text extraction."""
        if not image_data.startswith("data:"):
            image_data = f"data:image/png;base64,{image_data}"

        try:
            response = requests.post(
                f"{self.valves.vision_api_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.valves.vision_model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.valves.extraction_prompt},
                            {"type": "image_url", "image_url": {"url": image_data}}
                        ]
                    }],
                    "max_tokens": 1024
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Image processing error: {str(e)}]"

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")

        messages = body.get("messages", [])

        for msg in messages:
            content = msg.get("content")

            # Handle OpenAI vision format (list with text and image_url)
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item["text"])
                    elif item.get("type") == "image_url":
                        image_url = item["image_url"].get("url", "")
                        extracted = self._process_image(image_url)
                        text_parts.append(f"[Image content: {extracted}]")

                msg["content"] = "\n\n".join(text_parts)

            # Handle Ollama format (images array)
            if "images" in msg:
                extracted_texts = []
                for img in msg["images"]:
                    extracted = self._process_image(img)
                    extracted_texts.append(f"[Image content: {extracted}]")

                original = msg.get("content", "")
                msg["content"] = "\n\n".join(extracted_texts) + "\n\n" + original
                del msg["images"]

        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        return body
