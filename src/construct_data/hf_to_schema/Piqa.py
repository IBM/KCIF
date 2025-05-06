from datasets import load_dataset
from ..constants import create_schema, SCHEMA_KEYS


def transform_piqa(test_instance: dict):
    """Convert instance into the schema defined

    Args:
        test_instance (dict): each instance of the PIQA dataset

    Returns:
        dict: the provided instance in the schema defined
    """

    # initialize the schema for every instance
    schema = create_schema()
    schema[SCHEMA_KEYS.DATA_SET.value] = "Piqa"
    schema[SCHEMA_KEYS.HF_DATA_NAME.value] = (
        "https://huggingface.co/datasets/ybisk/piqa"
    )
    schema[SCHEMA_KEYS.TASK_TYPE.value] = "MCQ"

    # Copying untransformed data fields
    schema[SCHEMA_KEYS.INPUT_INSTANCE.value] = (
        f"Question: {test_instance['goal']}\nOptions:"
    )
    schema[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value] = [
        test_instance["sol1"],
        test_instance["sol2"],
    ]
    schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value] = ["A", "B"]

    if int(test_instance["label"]) == 0:
        schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_LABEL.value] = "A"
        schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value] = test_instance["sol1"]
        schema[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value] = ["A"]
    elif int(test_instance["label"]) == 1:
        schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_LABEL.value] = "B"
        schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value] = test_instance["sol2"]
        schema[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value] = ["B"]

    # The instruction prefix which gets appended
    schema[SCHEMA_KEYS.TASK_PROMPT.value] = (
        "Given a question and two answer candidates 'A' and 'B', "
    )
    schema[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] = (
        "Given a question and the possible answer candidates 'A' and 'B', answer the question by selecting the value associated with the option label corresponding to the correct answer.\n"
    )
    schema[SCHEMA_KEYS.FINAL_SUFFIX_TASK_INSTRUCTION.value] = "\n"

    schema[SCHEMA_KEYS.TASK_INSTRUCTIONS.value].append(
        schema[SCHEMA_KEYS.TASK_PROMPT.value]
    )

    assert schema[SCHEMA_KEYS.INPUT_INSTANCE.value] != ""
    assert schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value] != ""
    assert len(schema[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value]) == len(
        schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value]
    )
    assert schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_LABEL.value] != ""
    assert len(schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value]) > 0
    assert len(schema[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value]) > 0

    assert schema[SCHEMA_KEYS.TASK_PROMPT.value] != ""
    assert len(schema[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value]) > 0
    return schema


class Piqa:
    """
    This Class holds the Piqa dataset post transformation into the schema required
    """

    def __init__(self):
        """
        Initialize Piqa instance
        """
        super().__init__()
        # load the dataset
        self.dataset = load_dataset(
            "ybisk/piqa", split="validation", trust_remote_code=True
        )

        # convert the dataset into the schema defined
        self.intermediate_representation = self.dataset.map(
            transform_piqa, remove_columns=self.dataset.column_names
        )
