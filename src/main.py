import time
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

from attack.attack_modules_prompt_injection import PromptInjection
from defence.defence_modules_prompt_injection import InjectionDefence
from data.data_module import DataModule
from model.ollama_model import OllamaModel
from util.config import load_config
from util.seed import seed_everything
from util.evaluator import Evaluator

# TODO: measure ollama res consumption 
# prompt_eval_count → numero di token del prompt in input
# eval_count → numero di token generati in output
# prompt_eval_duration e eval_duration → tempi (in nanosecondi) per elaborare prompt e generazione
def main():
    cf = load_config("../ollama_config.yaml")
    parser = ArgumentParser(conflict_handler="resolve", add_help=True) 
    parser.add_argument(
        "--seed", type=int, default=cf["experiment"]["seed"], help='Seed for reproducibility')
    parser.add_argument(
        "-t", "--task", type=str, default=cf["experiment"]["task"],
        choices=["financial_sentiment", "twitter_news"], help='Task to perform')
    parser.add_argument(
        "-l", "--log_dir", type=str, default=cf["experiment"]["log_dir"],
        help='Path to save the output dataframe')
    parser.add_argument(
    "--run-id", type=str, default=None, help="Unique identifier for the experiment run")
    parser.add_argument(
        "-m", "--model", type=str, default=cf["model"]["name"], help='Model name to use')
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
        "-c", "--clean", action="store_true", default=False, 
        help='Run the model without any attack or defence')
    parser.add_argument(
        "--attack", type=str, default=cf["attack"]
    )
    parser.add_argument(
        "--defence", type=str, default=cf["defence"]
    )
    args = parser.parse_args()

    model_to_fn = {
        "llama3.1:8b" : "llama_3_1_8b",
        "gpt-oss:20b" : "gpt_oss_20b",
        "qwen3:0.6b" : "qwen_3_0_6b",
    }

    # Create a unique log directory for this run based on the current timestamp.
    base_log_dir = Path(args.log_dir).resolve()
    run_id = args.run_id if args.run_id else f'run_{round(time.time())}'
    log_dir_ver = base_log_dir / run_id
    log_dir_ver.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed, "../ollama_config.yaml")  # Set the random seed for reproducibility

    # Reading the instructions from the config file
    compl_gate_istr = cf[args.task]["compliance_gate_instruction"]
    task_instruction = cf[args.task]["instruction"]

    # Initialization
    dm = DataModule("../ollama_config.yaml", args.task)
    attack = PromptInjection(
        attack_name=args.attack, 
        injected_instruction=compl_gate_istr
    )
    defence = InjectionDefence(
        defence_name=args.defence, 
        instruction=task_instruction, 
        task=cf[args.task]["task_name"],
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
    evaluator = Evaluator(task=args.task, num_classes=dm.num_classes)

    # Loop through the dataset and process each prompt
    df_res = pd.DataFrame()
    for target_text, label in tqdm(dm.iter_all(), desc="Processing prompts", total=dm.size):
        
        if not args.clean:
            # Apply attack and defence mechanisms if not in clean mode
            attacked_text = attack.inject(target_text=target_text)
            final_prompt = defence.defend(attacked_text=attacked_text, clean_text=target_text)
        else:
            final_prompt = f"{target_text}\n\n{task_instruction}"  # No attack or defence, just use the original text #TODO: change target_text with target_text and instruction

        # TODO: measure reply time
        response = model.invoke(final_prompt)

        result = evaluator.evaluate(response=response, label=label, final_prompt=final_prompt)

        # TODO: measure TOKEN(added_defense_words)
        result["target_text"] = target_text
        df_res = pd.concat([df_res, pd.DataFrame([result])])

    # Stop the model after the test
    model.stop()

    # Save the results to a CSV file
    df_res.to_csv(f"{log_dir_ver}/{model_to_fn[args.model]}_{args.task}.csv")
    with open(f"{log_dir_ver}/dict_args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Results saved to {log_dir_ver}")

if __name__ == "__main__":
    main()