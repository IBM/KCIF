import random as r
from ..constants import SCHEMA_KEYS, CLASSIFICATION, COT_SUFFIX

seed = 2024
r.seed(seed)

# print_correct_answer
print_correct_answer_schema = {
    "instruction_id": "print_correct_answer",
    "instruction_text": [
        "print the text associated with the option label that answers the question correctly. Do not print the option label."
    ],
    "operation": [" CORRECT_ANSWER_TEXT "],
    "python_function_name": "print_correct_answer",
    "python_args": "instruction_instance",
    "target_answer_return_type": "text",
}


def print_correct_answer(input_instance: dict):
    """This function applies the print_correct_answer instruction on
    the given instance. In addition, it defines the reasoning and instruction
    following errors for the instruction

    Args:
        input_instance (dict): input instance

    Returns:
        dict: transformed instance
    """

    input_instance[SCHEMA_KEYS.INSTRUCTION_ID.value] = print_correct_answer_schema[
        "instruction_id"
    ]
    new_instruction = print_correct_answer_schema["instruction_text"][0]

    # we add reversing correct answer instruction to the instruction list
    input_instance[SCHEMA_KEYS.TASK_INSTRUCTIONS.value].append(new_instruction)

    input_instance[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] = (
        input_instance[SCHEMA_KEYS.TASK_PROMPT.value] + " " + new_instruction
    )
    input_instance[SCHEMA_KEYS.COT_INSTRUCTION.value] = (
        input_instance[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] + COT_SUFFIX
    )

    # return the correct answer text
    input_instance[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value].append(
        str(input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value])
    )
    input_instance[
        SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value
    ] += input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value]

    # apply instruction to all candidate outputs
    candidate_outputs = input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value]
    candidate_outputs_after_instruction = []
    for candidate_label, candidate in zip(
        input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value],
        candidate_outputs,
    ):
        if candidate != input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value]:
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(candidate)
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                candidate_label
            )
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                candidate_label + " " + candidate
            )
        else:
            input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
                candidate_label + " " + candidate
            )

    input_instance[CLASSIFICATION.CLASSIFICATION] = CLASSIFICATION.STRING_MANIPULATION
    return input_instance
