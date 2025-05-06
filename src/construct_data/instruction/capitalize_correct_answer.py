import random as r
from ..constants import SCHEMA_KEYS, CLASSIFICATION, COT_SUFFIX

seed = 2024
r.seed(seed)

# capitalize_correct_answer
capitalize_correct_answer_schema = [
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


def capitalize_correct_answer(input_instance: dict):
    """This function applies the capitalize_correct_answer instruction on
    the given instance. In addition, it defines the reasoning and instruction
    following errors for the instruction

    Args:
        input_instance (dict): input instance

    Returns:
        dict: transformed instance
    """

    assert input_instance[SCHEMA_KEYS.TASK_TYPE.value] == "MCQ"
    input_instance[SCHEMA_KEYS.INSTRUCTION_ID.value] = capitalize_correct_answer_schema[
        0
    ]["instruction_id"]
    new_instruction = capitalize_correct_answer_schema[0]["ext"][
        r.randint(0, len(capitalize_correct_answer_schema[0]["ext"]) - 1)
    ]

    # we add capitalizing correct answer instruction to the instruction list
    input_instance[SCHEMA_KEYS.TASK_INSTRUCTIONS.value].append(new_instruction)

    input_instance[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] = (
        input_instance[SCHEMA_KEYS.TASK_PROMPT.value] + " " + new_instruction
    )
    input_instance[SCHEMA_KEYS.COT_INSTRUCTION.value] = (
        input_instance[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] + COT_SUFFIX
    )
    input_instance[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value].append(
        input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value].upper()
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
        input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
            candidate_label.upper()
        )
        input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
            candidate_label.lower()
        )
        if candidate != input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value]:
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(candidate)
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                candidate.lower()
            )
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                candidate.upper()
            )
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                candidate_label.upper()
            )
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                candidate_label.lower()
            )
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                candidate_label
            )
            if (
                all(x.isalpha() or x.isspace() for x in candidate.strip())
                and len(candidate) > 1
            ):
                input_instance[
                    SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value
                ].append(candidate.lower())

        else:
            if (
                all(x.isalpha() or x.isspace() for x in candidate.strip())
                and len(candidate) > 1
            ):
                input_instance[
                    SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value
                ].append(candidate.lower())
    # input_instance["candidate_answer_instruction_output"].append(candidate_outputs_after_instruction)
    input_instance[CLASSIFICATION.CLASSIFICATION] = CLASSIFICATION.STRING_MANIPULATION
    return input_instance
