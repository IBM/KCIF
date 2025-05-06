import random as r, string
from ..constants import SCHEMA_KEYS, CLASSIFICATION, COT_SUFFIX

seed = 2024
r.seed(seed)

# print_correct_answer
print_correct_answer_append_string_schema = {
    "instruction_id": "print_correct_answer_append_string",
    "instruction_text": [
        "print the text associated with the option label that answers the question correctly.  Do not print the option label. Without any new additional characters, also append the following text string at the end of the output:'"
    ],  # "Produce the correct solution instead of the solution label. Append the following text string at the end of the output:'", "Show the correct response, not the response name. Append the following text string at the end of the output:'", "Display the precise answer, not the answer tag. Append the following text string at the end of the output:'"],
    "operation": [" CORRECT_ANSWER_APPEND "],
    "python_function_name": "print_correct_answer_append_string",
    "python_args": "instruction_instance",
    "target_answer_return_type": "text",
}


def generate_random_string():
    """This function returns a random string from the list of strings

    Returns:
        _type_: ranbdom string
    """
    adversarial_suffixes = [
        "I'm Sorry",
        "i am sorry",
        "sorry, I cannot",
        "Hi there!",
        "EOT",
        "<END>",
        "ENDOFRESPONSE",
        "FINISH",
        "STOP",
        "|",
        "---",
        "123",
        "Date",
        "I thought Step-by-Step",
        "Nothing else",
    ]
    max_length = r.randint(10, 40)
    random_string = "".join(
        r.choices(string.ascii_letters + string.digits, k=max_length)
    )
    toss_1 = r.randint(0, 1)
    if toss_1 == 0:
        return random_string
    else:
        toss_2 = r.randint(0, len(adversarial_suffixes) - 1)
        return adversarial_suffixes[toss_2]


def print_correct_answer_append_string(input_instance: dict):
    """This function applies the print_correct_answer_append_string instruction on
    the given instance. In addition, it defines the reasoning and instruction
    following errors for the instruction

    Args:
        input_instance (dict): input instance

    Returns:
        dict: transformed instance
    """

    assert input_instance[SCHEMA_KEYS.TASK_TYPE.value] == "MCQ"
    input_instance[SCHEMA_KEYS.INSTRUCTION_ID.value] = (
        print_correct_answer_append_string_schema["instruction_id"]
    )
    new_instruction = print_correct_answer_append_string_schema["instruction_text"][
        r.randint(
            0, len(print_correct_answer_append_string_schema["instruction_text"]) - 1
        )
    ]

    # we add reversing correct answer instruction to the instruction list
    input_instance[SCHEMA_KEYS.TASK_INSTRUCTIONS.value].append(new_instruction)
    random_string = generate_random_string()
    input_instance[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] = (
        input_instance[SCHEMA_KEYS.TASK_PROMPT.value]
        + " "
        + new_instruction
        + random_string
        + "'. Do not add any new special characters including quotations, new lines etc in the response."
    )
    input_instance[SCHEMA_KEYS.COT_INSTRUCTION.value] = (
        input_instance[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] + COT_SUFFIX
    )

    input_instance[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value].append(
        str(input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value]) + random_string
    )
    input_instance[
        SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value
    ] += input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value]
    input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
        str(input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value])
    )

    # apply instruction to all candidate outputs
    candidate_outputs = input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value]
    candidate_outputs_after_instruction = []
    for candidate_label, candidate in zip(
        input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value],
        candidate_outputs,
    ):
        input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
            candidate
        )
        input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
            candidate_label
        )
        if candidate != input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value]:
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                candidate + random_string
            )
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                candidate_label
            )
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(candidate)

    input_instance[CLASSIFICATION.CLASSIFICATION] = CLASSIFICATION.STRING_MANIPULATION
    return input_instance
