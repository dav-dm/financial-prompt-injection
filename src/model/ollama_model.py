import requests


class OllamaModel:
    def __init__(self, model_name, seed, temperature, log_probs, stream, num_predict):
        self.model_name = model_name
        self.seed = seed
        self.temperature = temperature
        self.log_probs = log_probs
        self.stream = stream
        self.num_predict = num_predict


    def invoke(self, prompt):
        if isinstance(prompt, str):
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": self.stream,
                "logprobs": self.log_probs,
                "options": {
                    "seed": self.seed,
                    "temperature": self.temperature,
                    "num_predict": self.num_predict
                }
            }
            req = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=120
            )
            req.raise_for_status()

            data = req.json()
            return data.get("response", "")
        
        elif isinstance(prompt, list):
            payload = {
                "model": self.model_name,
                "messages": prompt,
                "stream": self.stream,
                "logprobs": self.log_probs,
                "options": {
                    "seed": self.seed,
                    "temperature": self.temperature,
                    "num_predict": self.num_predict
                }
            }
            req = requests.post(
                "http://localhost:11434/api/chat",
                json=payload,
                timeout=120
            )
            req.raise_for_status()

            data = req.json()
            return data["message"]["content"]
        else:
            raise ValueError("Prompt must be a string or a list of messages.")


    def stop(self):
        requests.post(
        "http://localhost:11434/api/generate",
            json={
                "model": self.model_name,
                "keep_alive": 0
            },
            timeout=30
        ).raise_for_status()
