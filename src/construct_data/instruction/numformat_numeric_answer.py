import random as r
from ..constants import SCHEMA_KEYS, CLASSIFICATION, COT_SUFFIX

seed = 2024
r.seed(seed)
import locale

locale.setlocale(locale.LC_NUMERIC, "en_US.UTF-8")

# print_correct_answer_in_words
numformat_numeric_answer_schema = {
    "instruction_id": "numformat_numeric_answer",
    "instruction_text": [
        "print the text associated with the option label that answers the question correctly. If the answer is numeric print it in two decimal places as long as it contains no other string or units of measurement.  Do not print the option label.",
        "print the text associated with the option label that answers the question correctly. Numeric answer values should be printed in two decimal places as long as it contains no other string or units of measurement.  Do not print the option label.",
    ],
    "operation": [" NUMFORMAT_NUMERIC "],
    "python_function_name": "numformat_numeric_answer",
    "python_args": "instruction_instance",
    "target_answer_return_type": "text",
}


def apply_instruction(input_candidate: str):
    """This function checks if the given text contains a numeric value or not.
    If the text contains a numeric value, then returns the numeric value as a string
    with decimal places

    Args:
        input_candidate (str): input text

    Returns:
        _type_: transformed text
    """
    successful = False
    input_candidate = input_candidate.replace(",", "")
    try:
        # only add instruction for numericals if answer is numerical. This will give an error and exit to except block for non numerics.
        instruction_float = locale.atof(input_candidate)
        successful = True
        return successful, f"{instruction_float:0.2f}"
    except Exception as e:
        return successful, input_candidate


def numformat_numeric_answer(input_instance: dict):
    """This function applies the numformat_numeric_answer instruction on
    the given instance. In addition, it defines the reasoning and instruction
    following errors for the instruction

    Args:
        input_instance (dict): input instance

    Returns:
        dict: transformed instance
    """

    assert input_instance[SCHEMA_KEYS.TASK_TYPE.value] == "MCQ"
    input_instance[SCHEMA_KEYS.INSTRUCTION_ID.value] = numformat_numeric_answer_schema[
        "instruction_id"
    ]
    new_instruction = numformat_numeric_answer_schema["instruction_text"][
        r.randint(0, len(numformat_numeric_answer_schema["instruction_text"]) - 1)
    ]
    input_instance[SCHEMA_KEYS.TASK_INSTRUCTIONS.value].append(new_instruction)
    # Always add instruction for numerical regardless of answer
    input_instance[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] = (
        input_instance[SCHEMA_KEYS.TASK_PROMPT.value] + " " + new_instruction
    )
    input_instance[SCHEMA_KEYS.COT_INSTRUCTION.value] = (
        input_instance[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] + COT_SUFFIX
    )

    input_instance[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value].append(
        input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value]
    )
    input_instance[
        SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value
    ] += input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value]
    successful, new_output = apply_instruction(
        input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value]
    )
    input_instance[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value].append(new_output)
    if successful:
        format_string = input_instance[
            SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value
        ].replace(",", "")
        input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
            f"{float(format_string):0.1f}"
        )
        input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
            f"{float(format_string):0.3f}"
        )
        input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
            f"{float(format_string):0.0f}"
        )
        input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
            f"{float(format_string):0.4f}"
        )

    # apply instruction to all candidate outputs
    candidate_outputs = input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value]
    candidate_outputs_after_instruction = []
    for candidate_label, candidate in zip(
        input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value],
        candidate_outputs,
    ):
        if candidate != input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value]:
            successful, new_output = apply_instruction(candidate)
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(new_output)
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                candidate_label
            )
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(candidate)
    input_instance[CLASSIFICATION.CLASSIFICATION] = CLASSIFICATION.NUMERIC_MANIPULATION
    return input_instance
