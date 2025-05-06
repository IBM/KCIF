import random as r
from ..constants import SCHEMA_KEYS, CLASSIFICATION, COT_SUFFIX

seed = 2024
r.seed(seed)

# reverse_correct_answer
reverse_correct_answer_schema = {
    "instruction_id": "reverse_correct_answer",
    "instruction_text": [
        "answer the question by printing the text associated with the correct option label, in reverse. Do not print the option label."
    ],  # "Perform a string reversal of the correct answer", "Reverse the sequence of characters in the correct answer."],
    "operation": [" REVERSE_CORRECT_ANSWER "],
    "python_function_name": "reverse_correct_answer",
    "python_args": "instruction_instance",
    "target_answer_return_type": "text",
}


def reverse_correct_answer(input_instance: dict):
    """This function applies the reverse_correct_answer instruction on
    the given instance. In addition, it defines the reasoning and instruction
    following errors for the instruction

    Args:
        input_instance (dict): input instance

    Returns:
        dict: transformed instance
    """

    assert input_instance[SCHEMA_KEYS.TASK_TYPE.value] == "MCQ"
    input_instance[SCHEMA_KEYS.INSTRUCTION_ID.value] = reverse_correct_answer_schema[
        "instruction_id"
    ]
    new_instruction = reverse_correct_answer_schema["instruction_text"][
        r.randint(0, len(reverse_correct_answer_schema["instruction_text"]) - 1)
    ]

    # we add reversing correct answer instruction to the instruction list
    input_instance[SCHEMA_KEYS.TASK_INSTRUCTIONS.value].append(new_instruction)
    input_instance[
        SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value
    ] += input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value]
    input_instance[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] = (
        input_instance[SCHEMA_KEYS.TASK_PROMPT.value] + " " + new_instruction
    )
    input_instance[SCHEMA_KEYS.COT_INSTRUCTION.value] = (
        input_instance[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] + COT_SUFFIX
    )
    if isinstance(input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value], list):
        input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
            str(input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value][0])
        )
        input_instance[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value].append(
            str(input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value][0][::-1])
        )
    else:

        input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
            str(input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value])
        )
        input_instance[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value].append(
            str(input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value][::-1])
        )

    # apply instruction to all candidate outputs
    candidate_outputs = input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value]
    gd = input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value]
    if isinstance(gd, list):
        gd = gd[0]
    for candidate_label, candidate in zip(
        input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value],
        candidate_outputs,
    ):
        if isinstance(candidate, list):
            candidate = candidate[0]
        if candidate != gd:
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(candidate)
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                candidate[::-1]
            )
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                candidate_label
            )

    input_instance[CLASSIFICATION.CLASSIFICATION] = CLASSIFICATION.STRING_MANIPULATION
    return input_instance
