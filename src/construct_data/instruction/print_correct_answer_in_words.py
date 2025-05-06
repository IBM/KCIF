import random as r
from ..constants import SCHEMA_KEYS, CLASSIFICATION, COT_SUFFIX
from num2words import num2words as nw

seed = 2024
r.seed(seed)

# print_correct_answer_in_words
print_correct_answer_in_words_schema = {
    "instruction_id": "print_correct_answer_in_words",
    "instruction_text": [
        "print the text associated with the option label that answers the question correctly.  However, if the correct answer is a numeric value with no additional text (including percentages, currency, units of measurement etc), print the numeric answer in words. For example, if the answer is '32' print 'thirty-two' without quotes. Do not print the option label.",
        "print the text associated with the option label that answers the question correctly. However, if the correct answer is a numeric value with no additional text (including percentages, currency, units of measurement etc), ensure that the numeric answer values are returned in words. For example, if the answer is '32' print 'thirty-two' without quotes. Do not print the option label.",
    ],  # "Display the precise answer, not the answer tag. If the answer is numeric value with no additional text (including percentages, currency, units of measurement etc), print the answer in words."],
    "operation": [" CORRECT_ANSWER_WORDS "],
    "python_function_name": "print_correct_answer_in_words",
    "python_args": "instruction_instance",
    "target_answer_return_type": "text",
}


def apply_instruction(input_candidate: str):
    """This function checks if the given input text contains any numeric value and
    retiurns the numeric value in words

    Args:
        input_candidate (str): input text

    Returns:
        _type_: ttransformed text
    """
    successful = False
    try:
        # we want to return any numerical value as a text
        numeral_answer_in_words = nw(input_candidate)
        successful = True
        return successful, numeral_answer_in_words
    except Exception as e:
        # we are returning the answer as is
        return successful, input_candidate


def print_correct_answer_in_words(input_instance: dict):
    """This function applies the print_correct_answer_in_words instruction on
    the given instance. In addition, it defines the reasoning and instruction
    following errors for the instruction

    Args:
        input_instance (dict): input instance

    Returns:
        dict: transformed instance
    """

    assert input_instance[SCHEMA_KEYS.TASK_TYPE.value] == "MCQ"

    input_instance[SCHEMA_KEYS.INSTRUCTION_ID.value] = (
        print_correct_answer_in_words_schema["instruction_id"]
    )
    new_instruction = print_correct_answer_in_words_schema["instruction_text"][
        r.randint(0, len(print_correct_answer_in_words_schema["instruction_text"]) - 1)
    ]

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
            if successful:
                input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(new_output)
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(candidate)
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                candidate_label
            )
    # input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(candidate_outputs_after_instruction)
    input_instance[CLASSIFICATION.CLASSIFICATION] = CLASSIFICATION.NUMERIC_MANIPULATION
    return input_instance
