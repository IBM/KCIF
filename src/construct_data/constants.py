from enum import Enum


class SCHEMA_KEYS(Enum):
    """
    Define SCHEMA_KEYS, an Enum of keys to be used in our schema
    """

    DATA_SET = "dataset"
    HF_DATA_NAME = "hf_dataset"
    TASK_TYPE = "task_type"
    INSTRUCTION_ID = "instruction_id"

    INSTANCE_ID = "instance_id"

    INPUT_INSTANCE = (
        "dataset_input"  # Instance given to the LLM without any instruction.
    )
    CANDIDATE_ANSWER_SET = (
        "candidate_answer_set"  # the list of all posssible answers for that instance
    )
    CANDIDATE_ANSWER_LABEL_SPACE = (
        "candidate_answer_label_space"  # the list of all posssible answer labels
    )
    GROUND_TRUTH_ANSWER_LABEL = "ground_truth_answer_label"
    GROUND_TRUTH_ANSWER_TEXT = "ground_truth_answer_text"

    TASK_PROMPT = "dataset_instruction"  # Task Prompt. Task prompt should not define how to generate the answer.
    FINAL_SUFFIX_TASK_INSTRUCTION = "final_suffix_task_instruction"  # The final task instruction which gets appended to the input and dataset_instruction
    FINAL_PREFIX_TASK_INSTRUCTION = "final_prefix_task_instruction"  # The final task instruction which gets prepended to the input and dataset_instruction

    TASK_INSTRUCTIONS = "task_instructions"
    INSTRUCTION_OUTPUT = "instruction_output"
    INSTRUCTION_FOLLOWING_ERRORS_SET = "instruction_following_errors_set"
    REASONING_ERROR_SET = "reasoning_error_set"
    COT_INSTRUCTION = "cot_instruction"


def create_schema():
    """
    Initialize the schema dictionary

    Returns:
        dict: schema
    """
    schema = {
        SCHEMA_KEYS.DATA_SET.value: "",
        SCHEMA_KEYS.HF_DATA_NAME.value: "",
        SCHEMA_KEYS.TASK_TYPE.value: "MCQ",
        SCHEMA_KEYS.INSTRUCTION_ID.value: "",
        SCHEMA_KEYS.INSTANCE_ID.value: "",
        SCHEMA_KEYS.INPUT_INSTANCE.value: "",
        SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value: [],
        SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value: [],
        SCHEMA_KEYS.GROUND_TRUTH_ANSWER_LABEL.value: "",
        SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value: "",
        SCHEMA_KEYS.TASK_PROMPT.value: "",
        SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value: "",
        SCHEMA_KEYS.FINAL_SUFFIX_TASK_INSTRUCTION.value: "",
        SCHEMA_KEYS.TASK_INSTRUCTIONS.value: [],
        SCHEMA_KEYS.INSTRUCTION_OUTPUT.value: [],
        SCHEMA_KEYS.INSTRUCTION_FOLLOWING_ERRORS_SET.value: [],
        SCHEMA_KEYS.REASONING_ERROR_SET.value: [],
        SCHEMA_KEYS.COT_INSTRUCTION.value: "",
    }

    return schema


class CLASSIFICATION:
    """
    Define all possible instruction categories in our KCIF
    """

    CLASSIFICATION = "CLASSIFICATION"
    STRING_MANIPULATION = "STRING_MANIPULATION"
    LABEL_MANIPULATION = "LABEL_MANIPULATION"
    NUMERIC_MANIPULATION = "NUMERIC_MANIPULATION"
    LIST_MANIPULATION = "LIST_MANIPULATION"
    LIST_STRING_MANIPULATION = "LIST_STRING_MANIPULATION"
    FORMATTING_LISTS = "FORMATTING_LISTS"

    def __init__(self):
        pass


COT_SUFFIX = " Think step by step and in the end, finish your response with 'Response:$RESPONSE' where $RESPONSE (without quotes) is the final output expected."


class CATEGORY_SHORTFORM:
    """
    Define short string names of all possible instruction categories in our KCIF
    """

    shortform = {
        "String Manipulation": "String",
        "Operations on List": "List Ops.",
        "Operations on List (Conditional)": "List Ops. Conditional",
        "Format Correct Answer": "Format Ans.",
        "No Manipulation": "None",
        "Numeric Manipulation": "Numeric",
        "print_correct_answer_label": "print_correct_answer_label",
        "print_correct_answer": "print_correct_answer",
    }

    def __init__(self):
        pass


class TABLE_ORDERING:
    """
    Defines ordering of instructions for computing and displaying the metrics
    """

    categories = [
        "print_correct_answer_label",
        "print_correct_answer",
        "Operations on List (Conditional)",
        "Operations on List",
        "String Manipulation",
        "Numeric Manipulation",
        "Format Correct Answer",
    ]

    categories = [
        "print_correct_answer_label",
        "print_correct_answer",
        "String Manipulation",
        "Numeric Manipulation",
        "Format Correct Answer",
        "Operations on List (Conditional)",
        "Operations on List",
    ]

    def __init__(self):
        pass


class CATEGORY_MAPPING:
    """
    Defines instruction to instruction category mapping
    """

    mapping = {
        "alternate_case_correct_answer": "String Manipulation",
        "capitalize_correct_answer": "String Manipulation",
        "reverse_correct_answer_alternate_case": "String Manipulation",
        "reverse_correct_answer": "String Manipulation",
        "numformat_numeric_answer": "Format Correct Answer",
        "print_correct_answer": "print_correct_answer",
        "print_correct_answer_in_words": "Format Correct Answer",
        "print_correct_answer_append_string": "Format Correct Answer",
        "print_correct_answer_label": "print_correct_answer_label",
        "increment_correct_numeric_answer_by_one": "Numeric Manipulation",
        "increment_incorrect_numeric_answers_by_one": "Operations on List (Conditional)",
        "sort_only_incorrect_answers": "Operations on List (Conditional)",
        "sort_options_to_create_string": "Operations on List",
        "use_incorrect_options_to_create_string": "Operations on List (Conditional)",
        "use_options_to_create_string": "Operations on List",
    }
    categories = [
        "String Manipulation",
        "Format Correct Answer",
        "No Manipulation",
        "Numeric Manipulation",
        "Operations on List",
        "Operations on List (Conditional)",
    ]

    def __init__(self):
        pass


class DATASETS:
    """
    Defines the datasets used in our KCIF
    """

    datasets = ["MMLUPro", "Piqa", "Winogrande", "BBH", "BoolQ", "MathQA"]

    def __init__(self):
        pass


class PAPER_TABLES:
    EXACT_MATCH_COLUMN_NAME = "em_after_response_strict"
    IF_ERROR_COLUMN_NAME = "instr_follow_error_strict"
    REASONING_ERROR_COLUMN_NAME = "reasoning_error_strict"

    def __init__(self):
        pass
