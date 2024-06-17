import argparse
from pathlib import Path

import torch
import datasets
from tqdm import tqdm

from src.core.train import get_model, get_tokenizer
from src.utils.conversation import Conversation
from src.utils.io import dump_json


@torch.inference_mode()
def run_eval(model_path, model_id, max_new_tokens, tokenizer_path: str = None):
    if not tokenizer_path:
        tokenizer_path = model_path
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = get_tokenizer(tokenizer_path)
    print(f"Loading model {model_id} from {model_path}")
    try:
        model = get_model(
            "auto",
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_impl="flash_attention_2",
        )
    except ValueError:
        model = get_model(
            "auto",
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_impl="eager",
        )
    model.cuda()
    model.eval()

    conv = Conversation()
    outputs = []
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    for example in tqdm(eval_set, desc="Eval"):
        conv.append_message(conv.roles[0], example["instruction"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt], return_tensors="pt").input_ids
        conv.clear_msg()
        # generate here is a placeholder for your models generations
        output_ids = model.generate(
            input_ids.cuda(),
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
        )
        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]) :]  # noqa: E203
        # be consistent with the template's stop_token_ids
        if conv.stop_token_ids:
            stop_token_ids_index = [
                i for i, id in enumerate(output_ids) if id in conv.stop_token_ids
            ]
            if len(stop_token_ids_index) > 0:
                output_ids = output_ids[: stop_token_ids_index[0]]

        output = tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        if conv.stop_str and isinstance(conv.stop_str, list):
            stop_str_indices = sorted(
                [
                    output.find(stop_str)
                    for stop_str in conv.stop_str
                    if output.find(stop_str) > 0
                ]
            )
            if len(stop_str_indices) > 0:
                output = output[: stop_str_indices[0]]
        elif conv.stop_str and output.find(conv.stop_str) > 0:
            output = output[: output.find(conv.stop_str)]

        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()

        if conv.name == "xgen" and output.startswith("Assistant:"):
            output = output.replace("Assistant:", "", 1).strip()

        example["output"] = output
        example["generator"] = model_id
        outputs.append(example)

    outpath = Path("results/alpaca_eval") / f"{model_id}.json"
    outpath.parent.mkdir(exist_ok=True, parents=True)
    dump_json(outputs, outpath, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=False,
        help="A custom name or path to tokenizer dir.",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )

    args = parser.parse_args()

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        max_new_tokens=args.max_new_token,
        tokenizer_path=args.tokenizer_path,
    )
