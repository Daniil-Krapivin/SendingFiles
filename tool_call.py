"""
  title: Qwen Vision Preprocessor
  description: Extract image content with Qwen2.5-VL before main LLM
  author: user
  version: 0.1.0
  license: MIT
  """

  from pydantic import BaseModel, Field
  import requests

  class Filter:
      class Valves(BaseModel):
          vision_api_url: str = Field(
              default="http://vllm-qwen:8000/v1",
              description="Qwen vLLM server URL"
          )
          vision_model: str = Field(
              default="Qwen/Qwen2.5-VL-3B-Instruct",
              description="Vision model name"
          )

      def __init__(self):
          self.valves = self.Valves()

      def inlet(self, body: dict, __user__: dict = None) -> dict:
          """Process images before sending to LLM."""

          messages = body.get("messages", [])

          for msg in messages:
              content = msg.get("content")

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

              if "images" in msg:
                  extracted_texts = []
                  for img in msg["images"]:
                      extracted = self._process_image(img)
                      extracted_texts.append(f"[Image content: {extracted}]")

                  msg["content"] = "\n\n".join(extracted_texts) + "\n\n" + msg.get("content", "")
                  del msg["images"]

          return body

      def _process_image(self, image_data: str) -> str:
          if not image_data.startswith("data:"):
              image_data = f"data:image/png;base64,{image_data}"

          try:
              response = requests.post(
                  f"{self.valves.vision_api_url}/chat/completions",
                  json={
                      "model": self.valves.vision_model,
                      "messages": [{
                          "role": "user",
                          "content": [
                              {"type": "text", "text": "Extract all text from this image. If no text, briefly describe it."},
                              {"type": "image_url", "image_url": {"url": image_data}}
                          ]
                      }],
                      "max_tokens": 1024
                  },
                  timeout=60
              )
              return response.json()["choices"][0]["message"]["content"]
          except Exception as e:
              return f"[Error: {e}]"
