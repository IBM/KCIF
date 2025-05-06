import json
import os
import torch
import logging
from glob import glob
from argparse import ArgumentParser, Namespace

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Unable tp Import VLLM")


logger = logging.getLogger(name="KCIF")


def get_args() -> Namespace:
    """Define Arguments used in command-line

    Returns:
        Namespace: _description_
    """
    parser = ArgumentParser()

    group = parser.add_argument_group("Evaluate LLM on KCIF")
    group.add_argument("--engine", default="vllm", type=str)
    group.add_argument("--base_url", default="", type=str)
    group.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="model name to use",
    )
    group.add_argument("--tokenizer_name", default="auto", type=str)

    group.add_argument("--tensor_parallel_size", type=int, default=1)
    group.add_argument("--dtype", type=str, default="auto")

    group.add_argument("--input_path", type=str, help="data path")

    group.add_argument(
        "--batch_size", type=int, default=1, help="batch size for inference"
    )

    group.add_argument(
        "--max_input_length", type=int, default=8192, help="max input length"
    )
    group.add_argument(
        "--max_output_length", type=int, default=256, help="max output length"
    )

    group.add_argument("--output_folder", type=str, help="output folder")

    args = parser.parse_args()
    return args


def apply_chat_template(user_message, tokenizer):
    """Applies chat template dfined by the tokenizer

    Args:
        user_message (_type_): user message
        tokenizer (_type_): tokenizer

    Returns:
        list: user message after applying chat template
    """
    chat = []

    conversation = {}
    conversation["role"] = "user"
    conversation["content"] = user_message

    chat.append(conversation)

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        prompt = user_message

    return prompt


def main() -> None:
    args = get_args()

    model = args.model_name

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    output_folder = args.output_folder
    print("loading model!")

    if args.tokenizer_name == "auto":
        args.tokenizer_name = args.model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.engine == "vllm":
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=args.max_output_length, n=1
        )
        llm = LLM(
            model=model,
            tokenizer=args.tokenizer_name,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=torch.bfloat16,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )
    elif args.engine == "hf":
        logger.warning(f"Loading huggingface model: {model}")
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            force_download=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = model.to(device)
    elif args.engine == "online":
        client = OpenAI(
            # Or use the `OPENAI_BASE_URL` env var
            base_url=args.base_url,
        )

    all_datasets = glob(os.path.join(args.input_path, "*jsonl"))
    all_results = []
    id_to_instance_mapping = {}

    # for each dataset and instruction combination
    for each_dataset_instruction in tqdm(all_datasets):
        logger.warning(f"Loading evaluation on {each_dataset_instruction}...")

        data_instances = []
        with open(
            each_dataset_instruction, "r", errors="ignore", encoding="utf8"
        ) as reader:
            for each_line in reader:
                instance = json.loads(each_line)
                instance["id"] = len(data_instances)
                data_instances.append(instance)
            reader.close()

        logger.info(
            f"Number of instances in dataset {each_dataset_instruction} is {len(data_instances)}"
        )

        if args.engine == "vllm":
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

            batch_examples = []
            for data_index in tqdm(
                range(len(data_instances)),
                desc=f"For each instance in dataset {each_dataset_instruction}",
            ):
                batch_examples.append(data_instances[data_index])

                if len(batch_examples) > args.batch_size:
                    input_prompts = []
                    # apply chat template for each example
                    for each_example in batch_examples:
                        input_prompts.append(
                            apply_chat_template(each_example["input"], tokenizer)
                        )

                    outputs = llm.generate(
                        input_prompts, sampling_params=sampling_params, use_tqdm=False
                    )

                    for each_index in range(len(batch_examples)):
                        batch_examples[each_index]["output"] = (
                            outputs[each_index].outputs[0].text
                        )
                        batch_examples[each_index]["prompt"] = input_prompts[each_index]
                        all_results.append(batch_examples[each_index])

                    batch_examples = []
        # Huggingface inference
        elif args.engine == "hf":
            batch_examples = []
            for data_index in tqdm(
                range(len(data_instances)),
                desc=f"For each instance in dataset {each_dataset_instruction}",
            ):
                batch_examples.append(data_instances[data_index])

                if len(batch_examples) > args.batch_size:
                    input_prompts = []
                    # apply chat template for each example
                    for each_example in batch_examples:
                        input_prompts.append(
                            apply_chat_template(each_example["input"], tokenizer)
                        )

                    inputs = tokenizer(input_prompts, return_tensors="pt", padding=True)
                    for key in inputs:
                        inputs[key] = inputs[key].to(device)

                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        do_sample=False,
                        num_return_sequences=1,
                        max_new_tokens=args.max_output_length,
                    )

                    decoded_outputs = [
                        tokenizer.decode(y, skip_special_tokens=False) for y in outputs
                    ]

                    for index in range(len(batch_examples)):
                        batch_examples[index]["output"] = decoded_outputs[index].split(
                            input_prompts[index]
                        )[-1]
                        temp = tokenizer.encode(
                            batch_examples[index]["output"], add_special_tokens=False
                        )
                        batch_examples[index]["output"] = tokenizer.decode(
                            temp, skip_special_tokens=True
                        )
                        all_results.append(batch_examples[index])

                    batch_examples = []
        elif args.engine == "online":
            for data_index in tqdm(
                range(len(data_instances)),
                desc=f"For each instance in dataset {each_dataset_instruction}",
            ):
                instance_id = (
                    str(data_instances[data_index]["id"])
                    + "$"
                    + data_instances[data_index]["dataset"]
                    + "$"
                    + data_instances[data_index]["instruction_id"]
                )
                if instance_id in id_to_instance_mapping:
                    all_results.append(data_instances[data_index])
                    continue

                # apply chat template for each example
                input_prompt = apply_chat_template(each_example["input"])
                chat_completion = client.chat.completions.create(
                    messages=input_prompt,
                    model=args.model_name,
                )

                data_instances[data_index]["output"] = input_prompt
                data_instances[data_index]["prompt"] = chat_completion.choices[
                    0
                ].text.strip()
                all_results.append(data_instances[data_index]["prompt"])

        with open(os.path.join(output_folder, f"all_results.jsonl"), "w") as writer:
            for each_result in all_results:
                writer.write(json.dumps(each_result))
                writer.write("\n")
            writer.close()


if __name__ == "__main__":
    main()
