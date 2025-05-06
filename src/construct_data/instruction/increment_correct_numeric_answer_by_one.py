import random as r
from ..constants import SCHEMA_KEYS, CLASSIFICATION, COT_SUFFIX

seed = 2024
r.seed(seed)

# print_correct_answer_in_words
increment_correct_numeric_answer_by_one_schema = {
    "instruction_id": "increment_correct_numeric_answer_by_one",
    "instruction_text": [
        "print the text associated with the option label that answers the question correctly. Note that if the correct answer is a numeric quanity, including dollar values and percentages but contains no other string or units of measurement, print the value after increasing its value by 1. Dollar values should be prefixed with '$'. Do not print the option label."
    ],  # "print the correct answer to the question, instead of the answer label. if the answer is a dollar value, or a percentage amount and contains no other string or units of measurement. Dollar values should be prefixed with '$' "],
    "operation": [" INCREMENT_CORRECT_ANSWER_BY_ONE "],
    "python_function_name": "increment_correct_numeric_answer_by_one",
    "python_args": "instruction_instance",
    "target_answer_return_type": "text",
}


def apply_instruction(input_candidate: str):
    """The function checks if the provided text contains a numeric value.
    If the text is a numeric value then increments the numeric value by one and
    returns the updated text.

    Args:
        input_candidate (str): input text

    Returns:
        _type_: _description_
    """
    successful = False
    dollar = False
    percent = False
    numeric_candidate = input_candidate
    numeric_candidate = numeric_candidate.replace(",", "")
    # should be just one dollar value
    if (input_candidate.startswith("$") or input_candidate.endswith("$")) and len(
        input_candidate.split("$")
    ) < 3:
        dollar = True
        numeric_candidate = numeric_candidate.replace("$", "")
    # should be just one % value
    elif input_candidate.endswith("%") and len(input_candidate.split("%")) < 3:
        percent = True
        numeric_candidate = numeric_candidate.replace("%", "")
    try:
        # we want to return any numerical value as a text
        try:
            numeric_answer = int(numeric_candidate)
        except Exception:
            numeric_answer = float(numeric_candidate)
        successful = True
        format_string = f"{numeric_answer+1:,}"
        if dollar:
            format_string = "$" + format_string
        elif percent:
            format_string = format_string + "%"
        return successful, format_string
    except Exception:
        # we are returning the answer as is
        return successful, input_candidate


def increment_correct_numeric_answer_by_one(input_instance: dict):
    """This function applies the increment_correct_numeric_answer_by_one instruction on
    the given instance. In addition, it defines the reasoning and instruction
    following errors for the instruction

    Args:
        input_instance (dict): input instance

    Returns:
        dict: transformed instance
    """

    assert input_instance[SCHEMA_KEYS.TASK_TYPE.value] == "MCQ"
    input_instance[SCHEMA_KEYS.INSTRUCTION_ID.value] = (
        increment_correct_numeric_answer_by_one_schema["instruction_id"]
    )
    new_instruction = increment_correct_numeric_answer_by_one_schema[
        "instruction_text"
    ][
        r.randint(
            0,
            len(increment_correct_numeric_answer_by_one_schema["instruction_text"]) - 1,
        )
    ]

    input_instance[SCHEMA_KEYS.TASK_INSTRUCTIONS.value].append(new_instruction)

    input_instance[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] = (
        input_instance[SCHEMA_KEYS.TASK_PROMPT.value] + " " + new_instruction
    )
    input_instance[SCHEMA_KEYS.COT_INSTRUCTION.value] = (
        input_instance[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] + COT_SUFFIX
    )

    # return the correct answer text
    input_instance[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value].append(
        input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value]
    )

    # Apply instruction
    successful, new_output = apply_instruction(
        input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value]
    )
    input_instance[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value].append(new_output)
    input_instance[
        SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value
    ] += input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value]
    if successful:
        input_instance[
            SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value
        ] += input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value]

    # apply instruction to all candidate outputs
    candidate_outputs = input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value]
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
            input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
                candidate_label
            )
    # input_instance["candidate_answer_instruction_output"].append(candidate_outputs_after_instruction)
    input_instance[CLASSIFICATION.CLASSIFICATION] = CLASSIFICATION.NUMERIC_MANIPULATION
    return input_instance
