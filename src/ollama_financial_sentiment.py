from argparse import ArgumentParser

from attack.attack_modules_prompt_injection import PromptInjection
from defence.defence_modules_prompt_injection import InjectionDefence
from data.data_module import DataModule
from model.ollama_model import OllamaModel
from util.config import load_config


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
    parser.add_argument(
        "--attack", type=str, default=cf["attack"]
    )
    parser.add_argument(
        "--defence", type=str, default=cf["defence"]
    )
    args = parser.parse_args()

    
    compl_gate_istr = cf["financial_sentiment"]["compliance_gate_instruction"]
    sentiment_istr = cf["financial_sentiment"]["sentiment_instruction"]

    dm = DataModule("../ollama_config.yaml", "financial_sentiment")
    attack = PromptInjection(
        attack_name=args.attack, 
        injected_instruction=compl_gate_istr
    )
    defence = InjectionDefence(
        defence_name=args.defence, 
        instruction=sentiment_istr, 
        task='Financial Sentiment Analysis',
        model_name=args.model,
        seed=args.seed,
    )
    model = OllamaModel(
        model_name=args.model,
        seed=args.seed,
        temperature=args.temperature,
        log_probs=args.log_probs,
        stream=args.stream,
        num_predict=args.num_predict
    )
    
    for target_text, label in dm.iter_train():
        print(target_text, label)
        input()
        attacked_text = attack.inject(target_text=target_text)
        print(attacked_text)
        input()
        final_prompt = defence.defence(attacked_text=attacked_text)
        print(final_prompt)
        input()
        response = model.invoke(final_prompt)
        print(response)
        break
    # Stop the model after the test
    model.stop()

if __name__ == "__main__":
    main()