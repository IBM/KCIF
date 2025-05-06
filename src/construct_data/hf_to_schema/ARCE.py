from datasets import load_dataset
from ..constants import create_schema, SCHEMA_KEYS


def get_choices(options):
    """This functions returns the possible candidate options and their corresponding values

    Args:
        options (_type_): Dictionary containing options and labels

    Returns:
        _type_: Candidate options, Candidate values, Dictionary mapping the two
    """
    candidate_set = []
    candidate_labels = []
    candidate_set = options["text"]
    candidate_labels = options["label"]

    key_dict = {}
    for text, label in zip(candidate_set, candidate_labels):
        key_dict[label] = text

    return candidate_set, candidate_labels, key_dict


def transform_arce(test_instance: dict):
    """Convert instance into the schema defined

    Args:
        test_instance (dict): each instance of the ARC-Challenge dataset

    Returns:
        dict: the provided instance in the schema defined
    """

    # initialize the schema for every instance
    schema = create_schema()
    schema[SCHEMA_KEYS.DATA_SET.value] = "ARCE"
    schema[SCHEMA_KEYS.HF_DATA_NAME.value] = (
        "https://huggingface.co/datasets/allenai/ai2_arc"
    )
    schema[SCHEMA_KEYS.TASK_TYPE.value] = "MCQ"

    # Adding meta data information
    schema[SCHEMA_KEYS.INSTANCE_ID.value] = test_instance["id"]

    # Copying untransformed data fields
    schema[SCHEMA_KEYS.INPUT_INSTANCE.value] = (
        f"Question: {test_instance['question']}\n"
    )

    candidate_set, candidate_labels, key_dict = get_choices(test_instance["choices"])

    schema[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value] = candidate_set
    schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value] = candidate_labels

    schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_LABEL.value] = [test_instance["answerKey"]]
    schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value] = key_dict[
        test_instance["answerKey"]
    ]

    schema[SCHEMA_KEYS.TASK_PROMPT.value] = (
        "Given a question and "
        + str(len(schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value]))
        + " options: "
        + ", ".join(schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value])
        + " as candidate answers, "
    )

    schema[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] = (
        schema[SCHEMA_KEYS.TASK_PROMPT.value]
        + "answer the question by selecting the value associated with the option label corresponding to the correct answer.\n"
    )

    schema[SCHEMA_KEYS.FINAL_SUFFIX_TASK_INSTRUCTION.value] = "\n"

    schema[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value] = [test_instance["answerKey"]]

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


class ARCE:
    """
    This Class holds the ARC-Easy dataset post transformation into the schema required
    """

    def __init__(self):
        """
        Initialize the ARCE instance
        """
        super().__init__()
        # load the dataset
        self.dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")

        # convert the dataset into the schema defined
        self.intermediate_representation = self.dataset.map(
            transform_arce, remove_columns=self.dataset.column_names
        )
