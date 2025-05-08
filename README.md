# Evaluating the Instruction-following Abilities of Language Models using Knowledge Tasks

This repository is the official implementation of [Evaluating the Instruction-following Abilities of Language Models using Knowledge Tasks](https://arxiv.org/abs/2410.12972)

KCIF is a benchmark for evaluating the instruction-following capabilities of Large Language Models (LLM). We adapt existing knowledge benchmarks and augment them with instructions that are a) conditional on correctly answering the knowledge task or b) use the space of candidate options in multiple-choice knowledge-answering tasks. KCIF allows us to study model characteristics, such as their change in performance on the knowledge tasks in the presence of answer-modifying instructions and distractor instructions.

## Getting Started

### Dependencies

* Python > 3.10 is preferred

### Installing

```bash
conda create -n KCIF python=3.10
conda activate KCIF
pip install -r requirements.txt
```
## Supported Datasets

- [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
- [MathQA](https://huggingface.co/datasets/allenai/math_qa)
- [BoolQ](https://huggingface.co/datasets/google/boolq)
- [PiQA](https://huggingface.co/datasets/ybisk/piqa)
- [Winogrande](https://huggingface.co/datasets/allenai/winogrande)
- More tasks will be added soon. 

## Evaluation
To evaluate any model on KCIF, pls run the following command

```bash
python inference/run_inference.py --engine <engine_type> --model_name <HF model name or local checkpoint> --input_path <path to KCIF> --batch_size <batch size> --output_folder <path to output folder>
```

For the full list of arguments pls run
```bash
python inference/run_inference.py --help
```

### Scoring Generations
The evaluation script expects a JSON configuration file containing paths to generations for both instruction-following (if_filepath) and non-instruction-following (noif_filepath) versions of each model.

Here's a sample json file
```json
[
    {
        "if_filepath": "test_results/Meta-Llama-3.1-8B-Instruct_instruction_follow/all_results.jsonl",
        "noif_filepath": "test_results/Meta-Llama-3.1-8B-Instruct_no_instruction_follow/all_results.jsonl",
        "model": "llama"
    },
    {
        "if_filepath": "test_results/Qwen2.5-72B-Instruct_instruction_follow/all_results.jsonl",
        "noif_filepath": "test_results/Qwen2.5-72B-Instruct_no_instruction_follow/all_results.jsonl",
        "model": "qwen_72B"
    }
]
```

Each entry in the config includes:
- if_filepath: Path to instruction-following generations
- noif_filepath: Path to non-instruction-following generations
- model: Name or identifier for the model.

### Usage
Run the following command to compute metrics based on the config:

```bash
python -m evaluation.compute_metrics --config path/to/config.json --output_folder path/to/output
```

Arguments:
- `--config`: Path to the configuration JSON file.
- `--output_folder`: Directory where the computed metrics will be saved.

Sample config file is provided in [sample/sample_lite_benchmark.json](sample/sample_lite_benchmark.json)


## Results
The following Table lists the performance of several LLMs on our leaderboard.

### Full Benchmark

| **Models**         | **μₑₘ** | **IC Score** | **KTS Score** | **μₑₘ′** | **Average Score** |
|--------------------|---------|--------------|---------------|----------|-------------------|
| Qwen2.5-72B        | 0.4488  | 0.5077       | 0.4708        | 0.6218   | 0.5123            |
| Qwen2.5-32B        | 0.419   | 0.4736       | 0.4519        | 0.6351   | 0.4949            |
| Llama-3.1-70B      | 0.3697  | 0.3735       | 0.3925        | 0.6109   | 0.4366            |
| Gemma-2-27B        | 0.3622  | 0.3783       | 0.3984        | 0.5177   | 0.4142            |
| Qwen2.5-14B        | 0.2819  | 0.3521       | 0.305         | 0.4443   | 0.3458            |
| Phi-3-medium       | 0.2589  | 0.2632       | 0.2799        | 0.4897   | 0.3229            |
| Gemma-2-9B         | 0.2417  | 0.2701       | 0.2688        | 0.484    | 0.3162            |
| Qwen2.5-7B         | 0.1921  | 0.2393       | 0.2           | 0.4061   | 0.2594            |
| Llama-3.1-8B       | 0.1646  | 0.1917       | 0.1773        | 0.3907   | 0.2311            |
| Phi-3-small        | 0.1472  | 0.1474       | 0.1686        | 0.3376   | 0.2002            |
| Qwen2.5-3B         | 0.1277  | 0.1341       | 0.1386        | 0.3021   | 0.1756            |
| Llama-3.2-3B       | 0.0946  | 0.0874       | 0.1021        | 0.2395   | 0.1309            |
| Phi-3.5-mini       | 0.0966  | 0.1179       | 0.1014        | 0.2044   | 0.1301            |
| Mistral-7B         | 0.0484  | 0.059        | 0.057         | 0.2451   | 0.1024            |
| Qwen2.5-1.5B       | 0.0382  | 0.0346       | 0.0435        | 0.1461   | 0.0656            |
| Llama-3.2-1B       | 0.0153  | 0.012        | 0.0176        | 0.0897   | 0.0337            |

**Table**: Performance of the Small, Medium, and Large Models on our Full Benchmark — models ranked in order of performance using the average score (higher is better).

### Lite Benchmark

| **Models**        | **μₑₘ** | **IC Score** | **KTS Score** | **μₑₘ′** | **Average Score** |
|-------------------|---------|--------------|---------------|----------|-------------------|
| GPT-4o-2024-08-06 | 0.5065  | 0.5174       | 0.5874        | 0.6889   | 0.575             |
| Llama-3.1-405B    | 0.4617  | 0.4888       | 0.5351        | 0.6387   | 0.5311            |
| Qwen2.5-72B       | 0.4348  | 0.5035       | 0.493         | 0.5768   | 0.502             |
| Qwen2.5-32B       | 0.409   | 0.4751       | 0.4755        | 0.5873   | 0.4867            |
| Llama-3.1-70B     | 0.3708  | 0.4138       | 0.4319        | 0.5645   | 0.4453            |
| GPT-4o-mini       | 0.394   | 0.4029       | 0.4689        | 0.4609   | 0.4317            |
| Gemma-2-27B       | 0.3497  | 0.3972       | 0.4194        | 0.4505   | 0.4042            |
| Qwen2.5-14B       | 0.2764  | 0.3523       | 0.3272        | 0.4084   | 0.3411            |
| Phi-3-medium      | 0.2518  | 0.2869       | 0.3054        | 0.4238   | 0.317             |
| Gemma-2-9B        | 0.2381  | 0.2828       | 0.292         | 0.4428   | 0.3139            |
| Qwen2.5-7B        | 0.1944  | 0.2513       | 0.2275        | 0.3411   | 0.2536            |
| Llama-3.1-8B      | 0.174   | 0.2203       | 0.2048        | 0.3513   | 0.2376            |
| Phi-3-small       | 0.1555  | 0.1809       | 0.1921        | 0.3027   | 0.2078            |
| Mistral-7B        | 0.0577  | 0.0808       | 0.0768        | 0.205    | 0.1051            |

**Table**: Performance of the Medium, Large and Frontier Models on our Lite Benchmark — ranked in order of performance using the average score (higher is better).


## Contributions
This section provides instructions on how to contribute to the KCIF benchmark.

### Adding new dataset and Instructions

- To add new dataset, please follow the guidelines [here](src/construct_data/hf_to_schema/README.md)
- To add a new instruction, please follow the guidelines [here](src/construct_data/instruction/README.md)
- To create the dataset for evaluation, create a `json` file with `dataset` names as the keys and `instructions` to be applied on the dataset as list of values
- A sample `json` file is provided [here](src/construct_data/config.json)
- Then run the following command
```bash
cd src
python construct_data/create_benchmark.py --config <path to json> --output_path <path to folder to store the dataset> --cot
```

### Todo

- [ ] Support new tasks (BBH, etc.)
- [ ] Add test cases 
- [ ] Support for OpenAI API


## Citation
If you find KCIF useful, please cite it as follows in your publication:

```bibtex
@misc{murthy2024evaluatinginstructionfollowingabilitieslanguage,
      title={Evaluating the Instruction-following Abilities of Language Models using Knowledge Tasks}, 
      author={Rudra Murthy and Prince Kumar and Praveen Venkateswaran and Danish Contractor},
      year={2024},
      eprint={2410.12972},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.12972}, 
}
```

