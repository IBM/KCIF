import random as r
from ..constants import SCHEMA_KEYS, CLASSIFICATION, COT_SUFFIX

seed = 2024
r.seed(seed)

# reverse correct answer and alternate case
reverse_correct_answer_alternate_case_schema = {
    "instruction_id": "reverse_correct_answer_alternate_case",
    "instruction_text": [
        "reverse the text associated with the answer label that correctly answers the question. Print this reversed text in alternate case starting with upper case.  Do not print the option label."
    ],
    "operation": [" REVERSE_CORRECT_ANSWER_ALT_CASE "],
    "python_function_name": "reverse_correct_answer_alternate_case",
    "python_args": "instruction_instance",
    "target_answer_return_type": "text",
}


def reverse_correct_answer_alternate_case(input_instance: dict):
    """This function applies the reverse_correct_answer_alternate_case instruction on
    the given instance. In addition, it defines the reasoning and instruction
    following errors for the instruction

    Args:
        input_instance (dict): input instance

    Returns:
        dict: transformed instance
    """

    assert input_instance[SCHEMA_KEYS.TASK_TYPE.value] == "MCQ"
    input_instance[SCHEMA_KEYS.INSTRUCTION_ID.value] = (
        reverse_correct_answer_alternate_case_schema["instruction_id"]
    )
    new_instruction = reverse_correct_answer_alternate_case_schema["instruction_text"][
        r.randint(
            0, len(reverse_correct_answer_alternate_case_schema["instruction_text"]) - 1
        )
    ]

    # we add reversing correct answer instruction to the instruction list
    input_instance[SCHEMA_KEYS.TASK_INSTRUCTIONS.value].append(new_instruction)

    # we add reversing correct answer instruction to the instruction list
    input_instance[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] = (
        input_instance[SCHEMA_KEYS.TASK_PROMPT.value] + " " + new_instruction
    )
    input_instance[SCHEMA_KEYS.COT_INSTRUCTION.value] = (
        input_instance[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] + COT_SUFFIX
    )

    # reverse the correct answer
    if isinstance(input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value], list):
        input_instance[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value].append(
            input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value][-1][::-1]
        )
    else:
        input_instance[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value].append(
            input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value][::-1]
        )

    # convert the correct answer into alternate case format
    alternate_case_correct_answer = [
        ele.upper() if not idx % 2 else ele.lower()
        for idx, ele in enumerate(
            input_instance[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value][-1]
        )
    ]
    input_instance[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value].append(
        "".join(alternate_case_correct_answer)
    )

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

        alternate_case_candidate_answer = [
            ele.upper() if not idx % 2 else ele.lower()
            for idx, ele in enumerate(candidate)
        ]
        # if and even length instruction then instruction is not invariant to order of reversing and case changes
        if (len(alternate_case_correct_answer) % 2 == 0) and all(
            x.isalpha() or x.isspace() for x in candidate
        ):
            input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
                "".join(alternate_case_candidate_answer)
            )
            input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
                "".join(reversed("".join(alternate_case_candidate_answer)))
            )

        if (
            all(x.isalpha() or x.isspace() for x in candidate)
            and len(candidate.strip()) > 1
        ):
            candidate = str(candidate[::-1])
            input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
                candidate
            )
            input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
                candidate[::-1]
            )
            input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
                candidate_label
            )

        if candidate != gd:
            alternate_case_correct_answer = [
                ele.upper() if not idx % 2 else ele.lower()
                for idx, ele in enumerate(candidate)
            ]

            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                "".join(alternate_case_correct_answer)
            )
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                "".join(reversed("".join(alternate_case_candidate_answer)))
            )
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(candidate)
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                candidate[::-1]
            )
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                candidate_label
            )

    input_instance[CLASSIFICATION.CLASSIFICATION] = CLASSIFICATION.STRING_MANIPULATION
    return input_instance
