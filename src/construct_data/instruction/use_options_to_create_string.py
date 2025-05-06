import random as r
import copy
from ..constants import SCHEMA_KEYS, CLASSIFICATION, COT_SUFFIX

seed = 2024
r.seed(seed)

# sort only incorrect answers; assuming no options are lists themeselves
use_options_to_create_string_schema = {
    "instruction_id": "use_options_to_create_string",
    "instruction_text": [
        "create a string by concatenating the last character of every option value (not option label). If the last character is a special character (such as period, comma, quotation, etc) use the previous character. "
    ],
    "operation": [" OPTIONS_TO_STRING "],
    "python_function_name": "use_options_to_create_string",
    "python_args": "instruction_instance",
    "target_answer_return_type": "list",
}


def apply_instruction(input_candidate: str, candidate_answer_set: list):
    """This function takes in a list as input, sorts the list in alphabetical order.
    The last alphanumeric character from each list item is extracted and concatenated
    to form a string

    Args:
        input_candidate (str): _description_
        candidate_answer_set (list): _description_

    Returns:
        _type_: _description_
    """

    return_str = ""
    for candidate in candidate_answer_set:
        chosen_char = "."
        index = -1
        while not chosen_char.isalnum() and len(candidate) >= abs(index):
            chosen_char = candidate[index]
            index -= 1
        return_str += chosen_char
    return return_str


def use_options_to_create_string(input_instance: dict):
    """This function applies the use_options_to_create_string instruction on
    the given instance. In addition, it defines the reasoning and instruction
    following errors for the instruction

    Args:
        input_instance (dict): input instance

    Returns:
        dict: transformed instance
    """

    assert input_instance[SCHEMA_KEYS.TASK_TYPE.value] == "MCQ"
    input_instance[SCHEMA_KEYS.INSTRUCTION_ID.value] = (
        use_options_to_create_string_schema["instruction_id"]
    )
    new_instruction = use_options_to_create_string_schema["instruction_text"][
        r.randint(0, len(use_options_to_create_string_schema["instruction_text"]) - 1)
    ]

    # we add reversing correct answer instruction to the instruction list
    input_instance[SCHEMA_KEYS.TASK_INSTRUCTIONS.value].append(new_instruction)

    input_instance[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] = (
        input_instance[SCHEMA_KEYS.TASK_PROMPT.value] + " " + new_instruction
    )
    input_instance[SCHEMA_KEYS.COT_INSTRUCTION.value] = (
        input_instance[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] + COT_SUFFIX
    )

    input_instance[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value].append(
        apply_instruction(
            input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value],
            input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value],
        )
    )

    # apply instruction to all candidate outputs
    candidate_outputs = input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value]
    input_instance[CLASSIFICATION.CLASSIFICATION] = CLASSIFICATION.LIST_MANIPULATION
    return input_instance
