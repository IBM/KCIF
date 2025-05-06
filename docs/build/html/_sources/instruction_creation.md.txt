# Instruction Creation

This readme contains information on adding new instructions.

## Adding new instruction
To add a new instruction, create a new instruction file. The instruction file should have the same name as the instruction name. For example, to add a instruction `capitalize_correct_answer` instruction, the filename should be `capitalize_correct_answer.py` and the file should have a function named `capitalize_correct_answer`. The function defines a Huggingface Dataset mapping transformation which applies instruction on each of the instance and creates an updated instance.

To begin with define an instruction schema which provides information around the instruction

```json
INFORMATION_SCHEMA=[
    {
        "instruction_id": "",
        "ext": [
            "",
            "",
        ],
        "operation": [""],
        "python_function_name": "",
        "python_args": "",
        "target_answer_return_type": "",
    },
]
```

Here,
- instruction_id: instruction name
- ext: list of instructions suffixes which is added to the task instruction
- operation: the instruction category
- python_function_name: Python function name
- python_args: the argument taken by the Python function
- target_answer_return_type: The return type of the answer post instruction application

For example, schema for `capitalize_correct_answer` looks like this

```json
capitalize_correct_answer_schema=[
    {
        "instruction_id": "capitalize_correct_answer",
        "ext": [
            "answer the question by printing the text associated with the correct option label in uppercase. Do not print the option label.",
            "capitalize the text associated with the optional label that answers the question correctly. Do not print the option label.",
        ],
        "operation": [" CAPITALIZE "],
        "python_function_name": "capitalize_correct_answer",
        "python_args": "instruction_instance",
        "target_answer_return_type": "text",
    },
]
```

The next step is to create a transformation function, which takes in a instance and adds transformation related to the instruction. 
The keys to override are
- `SCHEMA_KEYS.INSTRUCTION_ID.value` the instruction id or name being applied
- `SCHEMA_KEYS.TASK_INSTRUCTIONS.value` the instruction prompt gets appended to the list
- `SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value` the final instruction would concatenate the task prompt (`SCHEMA_KEYS.TASK_PROMPT.value`) with the instruction prompt
- `SCHEMA_KEYS.COT_INSTRUCTION.value` the COT instruction prompt would concatenate the `SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value` with the `COT_SUFFIX`
- `SCHEMA_KEYS.INSTRUCTION_OUTPUT.value` the output from applying the current instruction would be appended to the list
- `SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value` add possible instruction following errors as a list
- `SCHEMA_KEYS.REASONING_ERROR_SET.value` add possible reasoning errors as list