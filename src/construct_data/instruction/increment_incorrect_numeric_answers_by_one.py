import random as r
import copy, pdb
from ..constants import SCHEMA_KEYS, CLASSIFICATION, COT_SUFFIX

seed = 2024
r.seed(seed)

# print_correct_answer_in_words
increment_incorrect_numeric_answers_by_one_schema = {
    "instruction_id": "increment_incorrect_numeric_answers_by_one",
    "instruction_text": [
        "print the list of incorrect answers (not the answer label). Increase each value by 1 while printing if it is a numeric quanity including dollar values, percentages but contains no other string or units of measurement. Do not print the option labels. ",
        "print the list of incorrect answers all incremented by one if it is a numeric quanity including dollar values and percentages but contains no other string or units of measurement. Do not print the option labels. ",
    ],
    "operation": [" INCREMENT_INCORRECT_ANSWERS_BY_ONE "],
    "python_function_name": "increment_incorrect_numeric_answers_by_one",
    "python_args": "instruction_instance",
    "target_answer_return_type": "list",
}


def apply_instruction(input_candidate: str, candidate_answer_set: list):
    """
    Input candidate here is the "ground truth" that we want to avoid printing out.
    And then perform increment on all the other options in the candidate answer set
    """
    instruction_answer = []
    # we want to return any numerical value as a text
    templist = copy.deepcopy(candidate_answer_set)
    succesful = False
    try:
        templist.remove(input_candidate)
    except:
        return succesful, str(candidate_answer_set)

    for value in templist:
        dollar = False
        percent = False
        value_candidate = value
        value_candidate = value_candidate.replace(",", "")
        if value_candidate.startswith("$") and len(value_candidate.split("$")) < 3:
            dollar = True
            value_candidate = value_candidate.replace("$", "")
        # should be just one % value
        elif value_candidate.endswith("%") and len(value_candidate.split("%")) < 3:
            percent = True
            value_candidate = value_candidate.replace("%", "")

        try:
            try:
                numeric_answer = int(value_candidate)
            except Exception:
                numeric_answer = float(value_candidate)
            format_string = f"{numeric_answer+1:,}"
            if dollar:
                format_string = "$" + format_string
            elif percent:
                format_string = format_string + "%"
            succesful = True
            instruction_answer.append(str(format_string))
        except Exception as e:
            # we are returning the answer as is
            instruction_answer.append(str(value))

    return succesful, str(instruction_answer)


def increment_incorrect_numeric_answers_by_one(input_instance: dict):
    """This function applies the increment_incorrect_numeric_answers_by_one instruction on
    the given instance. In addition, it defines the reasoning and instruction
    following errors for the instruction

    Args:
        input_instance (dict): input instance

    Returns:
        dict: transformed instance
    """

    assert input_instance[SCHEMA_KEYS.TASK_TYPE.value] == "MCQ"
    input_instance[SCHEMA_KEYS.INSTRUCTION_ID.value] = (
        increment_incorrect_numeric_answers_by_one_schema["instruction_id"]
    )
    new_instruction = increment_incorrect_numeric_answers_by_one_schema[
        "instruction_text"
    ][
        r.randint(
            0,
            len(increment_incorrect_numeric_answers_by_one_schema["instruction_text"])
            - 1,
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

    # input_instance["candidate_answer_instruction_output"].append(input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value])
    successful, new_output = apply_instruction(
        input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value],
        input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value],
    )
    input_instance[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value].append(new_output)
    input_instance[
        SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value
    ] += input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value]

    # apply instruction to all candidate outputs
    candidate_outputs = input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value]

    if successful:
        templist = copy.deepcopy(candidate_outputs)
        templist.remove(input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value])
        input_instance[SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value].append(
            str(templist)
        )
    input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
        input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value]
    )

    for candidate_label, candidate in zip(
        input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value],
        candidate_outputs,
    ):
        if candidate != input_instance[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value]:
            successful, new_output = apply_instruction(
                candidate, input_instance[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value]
            )
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(new_output)
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(candidate)
            input_instance[SCHEMA_KEYS.REASONING_ERROR_SET.value].append(
                candidate_label
            )

    # input_instance["candidate_answer_instruction_output"].append(candidate_outputs_after_instruction)
    input_instance[CLASSIFICATION.CLASSIFICATION] = CLASSIFICATION.NUMERIC_MANIPULATION
    return input_instance
