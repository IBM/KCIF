# KCIF

## Evaluating the Instruction-following Abilities of Language Models using Knowledge Tasks

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
## Supported Tasks

- [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
- [MathQA](https://huggingface.co/datasets/allenai/math_qa)
- [BoolQ](https://huggingface.co/datasets/google/boolq)
- [PiQA](https://huggingface.co/datasets/ybisk/piqa)
- [Winogrande](https://huggingface.co/datasets/allenai/winogrande)
- More tasks will be added soon. 

## Usage

### Dataset Creation

- To add new dataset, please follow the guidelines [here](src/construct_data/hf_to_schema/README.md)
- To add a new instruction, please follow the guidelines [here](src/construct_data/instruction/README.md)
- To create the dataset for evalaution, create a `json` file with `dataset` names as the keys and `instructions` to be applied on the dataset as list of values
- A sample `json` file is provided [here](src/construct_data/config.json)
- Then run the following command
```bash
cd src
python construct_data/create_benchmark.py --config <path to json> --output_path <path to folder to store the dataset> --cot
```

### Evaluating Models
To evaluate any model on KCIF, pls run the following command

```bash
python inference/run_inference.py --engine <engine_type> --model_name <HF model name or local checkpoint> --input_path <path to KCIF> --batch_size <batch size> --output_folder <path to output folder>
```

For the full list of arguments pls run
```bash
python inference/run_inference.py --help
```

#### Scoring Generations
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

#### Usage
Run the following command to compute metrics based on the config:

```bash
python -m evaluation.compute_metrics --config path/to/config.json --output_folder path/to/output
```

Arguments:
- `--config`: Path to the configuration JSON file.
- `--output_folder`: Directory where the computed metrics will be saved.

Sample config file is provided in [sample/sample_lite_benchmark.json](sample/sample_lite_benchmark.json)

## Todo

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

