import random as r
import copy
from ..constants import SCHEMA_KEYS, CLASSIFICATION, COT_SUFFIX

seed = 2024
r.seed(seed)

# sort only incorrect answers; assuming no options are lists themeselves
sort_only_incorrect_answers_schema = {
    "instruction_id": "sort_only_incorrect_answers",
    "instruction_text": [
        "excluding the option that answers the question correctly, print a sorted list (ascending order) of the incorrect options. Do not print the option labels. Use the text associated with the option labels and not the option labels while sorting and printing.",
        "excluding the option that answers the question correctly, print a sorted list (ascending order) of the incorrect options.  Do not print the option labels. Use the text associated with the option labels and not the option labels while sorting and printing.",
        "excluding the option that answers the question correctly, print a sorted list (ascending order) of the incorrect options. Do not print the option labels. Use the text associated with the option labels and not the option labels while sorting and printing.",
    ],
    "operation": [" SORT_INCORRECT "],
    "python_function_name": "sort_only_incorrect_answers",
    "python_args": "instruction_instance",
    "target_answer_return_type": "list",
}


def apply_instruction(input_candidate: str, candidate_answer_set: list):
    """This function removes the given item from the list and returns the sorted list

    Args:
        input_candidate (str): item to be removed
        candidate_answer_set (list): list containing all item

    Returns:
        _type_: list with the given item removed
    """

    templist = copy.deepcopy(candidate_answer_set)
    templist.remove(input_candidate)
    templist.sort(reverse=False)
    return str(templist)


def sort_only_incorrect_answers(input_instance: dict):
    """This function applies the sort_only_incorrect_answers instruction on
    the given instance. In addition, it defines the reasoning and instruction
    following errors for the instruction

    Args:
        input_instance (dict): input instance

    Returns:
        dict: transformed instance
    """

    assert input_instance[SCHEMA_KEYS.TASK_TYPE.value] == "MCQ"
    input_instance[SCHEMA_KEYS.INSTRUCTION_ID.value] = (
        sort_only_incorrect_answers_schema["instruction_id"]
    )
    new_instruction = sort_only_incorrect_answers_schema["instruction_text"][
        r.randint(0, len(sort_only_incorrect_answers_schema["instruction_text"]) - 1)
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
