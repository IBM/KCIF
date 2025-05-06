Getting Started
===============

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

- To add new dataset, please follow the guidelines [here](data_creation_rst.rst)
- To add a new instruction, please follow the guidelines [here](instruction_creation_rst.rst)
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

## Todo

- [ ] Support new tasks (BBH, etc.)
- [ ] Add test cases 
- [ ] Support for OpenAI API
