import random as r
from ..constants import SCHEMA_KEYS, CLASSIFICATION, COT_SUFFIX, COT_SUFFIX

seed = 2024
r.seed(seed)

# alternate_case_correct_answer
alternate_case_correct_answer_schema = [
    {
        "instruction_id": "alternate_case_correct_answer",
        "instruction_text": [
            "answer the question by printing the text associated with the correct option label in alternate case. The first character should be in uppercase and the following characters should alternate between lowercase and uppercase. Do not print the option label."
        ],
        "operation": [" ALT_CASE "],
        "python_function_name": "alternate_case_correct_answer",
        "python_args": "instruction_instance",
        "target_answer_return_type": "text",
    },
]


def apply_instruction(input_candidate: str):
    """transforms the given string into alternate case. The first character will
    always be in uppercase, followed by lowercase, and then uppercase and so on

    Args:
        input_candidate (str): input string

    Returns:
        _type_: string in alternate case
    """
    alternate_case_correct_answer = [
        ele.upper() if not idx % 2 else ele.lower()
        for idx, ele in enumerate(input_candidate)
    ]
    alternate_case_correct_answer = "".join(alternate_case_correct_answer)
    return alternate_case_correct_answer


def apply_instruction_incorrect(input_candidate: str):
    """transforms the given string into alternate case. The first character will
    always be in lowecase, followed by uppercase, and the lowercase and so on

    Args:
        input_candidate (str): input string

    Returns:
        _type_: string in alternate case
    """
    alternate_case_correct_answer = [
        ele.upper() if idx % 2 else ele.lower()
        for idx, ele in enumerate(input_candidate)
    ]
    alternate_case_correct_answer = "".join(alternate_case_correct_answer)
    return alternate_case_correct_answer


def alternate_case_correct_answer(input_instance: dict):
    """This function applies the alternate_case_correct_answer instruction on
    the given instance. In addition, it defines the reasoning and instruction
    following errors for the instruction

    Args:
        input_instance (dict): input instance

    Returns:
        dict: transformed instance
    """
    assert input_instance[SCHEMA_KEYS.TASK_TYPE.value] == "MCQ"
    input_instance[SCHEMA_KEYS.INSTRUCTION_ID.value] = (
        alternate_case_correct_answer_schema[0][SCHEMA_KEYS.INSTRUCTION_ID.value]
    )
    new_instruction = alternate_case_correct_answer_schema[0]["instruction_text"][
        r.randint(
            0, len(alternate_case_correct_answer_schema[0]["instruction_text"]) - 1
        )
    ]

    # we add alternating correct answer instruction to the instruction list
    input_instance[SCHEMA_KEYS.TASK_INSTRUCTIONS.value].append(new_instruction)

    # we add alternating correct answer instruction to the instruction list
    input_instance[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] = (
        input_instance[SCHEMA_KEYS.TASK_PROMPT.value] + " " + new_instruction
    )
    input_instance[SCHEMA_KEYS.COT_INSTRUCTION.value] = (
        input_instance[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] + COT_SUFFIX
    )

    # convert output to alternate case
    instr_output_correct_answer = apply_instruction(
        input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value]
    )
    input_instance[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value].append(
        instr_output_correct_answer
    )
    input_instance[
        SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value
    ] += input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value]
    # input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value]+=input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value]

    input_instance[CLASSIFICATION.CLASSIFICATION] = CLASSIFICATION.STRING_MANIPULATION

    # convert candidate outputs to alternate case
    candidate_outputs = input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value]
    candidate_outputs_after_instruction = []
    for candidate_label, candidate in zip(
        input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value],
        candidate_outputs,
    ):
        input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
            apply_instruction_incorrect(candidate)
        )
        if candidate != input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value]:
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(candidate)
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                apply_instruction(candidate)
            )
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                candidate_label
            )
        else:
            if (
                all(x.isalpha() or x.isspace() for x in candidate)
                and len(candidate) > 1
            ):
                input_instance[
                    SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value
                ].append(candidate.lower())
                input_instance[
                    SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value
                ].append(candidate.upper())
                input_instance[
                    SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value
                ].append(candidate)

    return input_instance
