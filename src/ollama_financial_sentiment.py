import requests
import pandas as pd
from argparse import ArgumentParser

from util.config import load_config
from data.data_module import DataModule


def main():
    cf = load_config("../ollama_config.yaml")

    parser = ArgumentParser(conflict_handler="resolve", add_help=True) 
    parser.add_argument(
        "--seed", type=int, default=cf["experiment"]["seed"], help='Seed for reproducibility')
    parser.add_argument(
        "--model", type=str, default=cf["model"]["name"], help='Model name to use')
    parser.add_argument(
        "--temperature", type=float, default=cf["model"]["temperature"], 
        help='Temperature for randomness in generation')
    parser.add_argument(
        "--log-probs", type=bool, default=cf["model"]["logprobs"],
        help='Whether to return log probabilities')
    parser.add_argument(
        "--stream", type=bool, default=cf["model"]["stream"],
        help='Whether to stream the response')
    parser.add_argument(
        "--num-predict", type=int, default=cf["model"]["num_predict"],
        help='Number of tokens to predict')
    
    args = parser.parse_args()

    # payload = {
    #     "model": args.model,
    #     "prompt": "Say hello and confirm which model are you.",
    #     "stream": args.stream,
    #     "logprobs": args.log_probs,
    #     "options": {
    #         "seed": args.seed,
    #         "temperature": args.temperature,
    #         "num_predict": args.num_predict
    #     }
    # }
    # response = requests.post(
    #     "http://localhost:11434/api/generate",
    #     json=payload,
    #     timeout=120
    # )
    # response.raise_for_status()

    # data = response.json()
    # # print(json.dumps(data, indent=2, ensure_ascii=False))
    # print("RESPONSE:", repr(data.get("response", "")))

    # # Stop the model after the test
    # requests.post(
    #     "http://localhost:11434/api/generate",
    #     json={
    #         "model": args.model,
    #         "keep_alive": 0
    #     },
    #     timeout=30
    # ).raise_for_status()

    compl_gate_istr = cf["financial_sentiment"]["compliance_gate_instruction"]
    sentiment_istr = cf["financial_sentiment"]["sentiment_instruction"]

    dm = DataModule("../ollama_config.yaml", "financial_sentiment")
    


if __name__ == "__main__":
    main()