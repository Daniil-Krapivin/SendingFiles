 """
  title: Qwen Vision Preprocessor
  description: Extract image content with Qwen2.5-VL, send text to main LLM
  version: 0.1.0
  """

  import requests
  import base64
  from pydantic import BaseModel, Field
  from typing import Optional

  class Pipe:
      class Valves(BaseModel):
          vision_api_url: str = Field(
              default="http://localhost:8000/v1",
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

      def __init__(self):
          self.valves = self.Valves()

      def process_image(self, image_data: str) -> str:
          """Send image to Qwen and get text extraction."""

          # Ensure proper base64 format
          if not image_data.startswith("data:"):
              image_data = f"data:image/png;base64,{image_data}"

          try:
              response = requests.post(
                  f"{self.valves.vision_api_url}/chat/completions",
                  headers={"Content-Type": "application/json"},
                  json={
                      "model": self.valves.vision_model,
                      "messages": [
                          {
                              "role": "user",
                              "content": [
                                  {"type": "text", "text": self.valves.extraction_prompt},
                                  {"type": "image_url", "image_url": {"url": image_data}}
                              ]
                          }
                      ],
                      "max_tokens": 1024
                  },
                  timeout=60
              )
              response.raise_for_status()
              result = response.json()
              return result["choices"][0]["message"]["content"]
          except Exception as e:
              return f"[Image processing failed: {str(e)}]"

      def pipe(self, body: dict) -> dict:
          """Process images in messages before sending to main LLM."""

          messages = body.get("messages", [])

          for msg in messages:
              content = msg.get("content")

              # Handle list-style content (OpenAI vision format)
              if isinstance(content, list):
                  text_parts = []
                  for item in content:
                      if item.get("type") == "text":
                          text_parts.append(item["text"])
                      elif item.get("type") == "image_url":
                          image_url = item["image_url"].get("url", "")
                          extracted = self.process_image(image_url)
                          text_parts.append(f"[Extracted from image: {extracted}]")

                  # Replace with plain text
                  msg["content"] = "\n\n".join(text_parts)

              # Handle images array (Ollama format)
              if "images" in msg:
                  extracted_texts = []
                  for img in msg["images"]:
                      extracted = self.process_image(img)
                      extracted_texts.append(f"[Extracted from image: {extracted}]")

                  original_content = msg.get("content", "")
                  msg["content"] = "\n\n".join(extracted_texts) + "\n\n" + original_content
                  del msg["images"]

          return body
