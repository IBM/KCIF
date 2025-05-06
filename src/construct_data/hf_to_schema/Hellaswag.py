import locale

from ..constants import SCHEMA_KEYS, create_schema
from datasets import load_dataset


def get_choices(options):
    """This functions returns the possible candidate options and their corresponding values

    Args:
        options (_type_): Dictionary containing options and labels

    Returns:
        _type_: Candidate options, Candidate values, Dictionary mapping the two
    """
    new_labels = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]
    candidate_set = []
    candidate_set = options
    key_dict = {}
    new_labels = new_labels[: len(options)]
    candidate_labels = new_labels
    index = 0
    for new_label in new_labels:
        key_dict[str(index)] = new_label
        index += 1
    return candidate_set, candidate_labels, key_dict


def transform_hellaswag(test_instance: dict):
    """Convert instance into the schema defined

    Args:
        test_instance (dict): each instance of the Hellaswag dataset

    Returns:
        dict: the provided instance in the schema defined
    """
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

    # initialize the schema for every instance
    schema = create_schema()
    schema[SCHEMA_KEYS.DATA_SET.value] = "Hellaswag"
    schema[SCHEMA_KEYS.HF_DATA_NAME.value] = (
        "https://huggingface.co/datasets/Rowan/hellaswag/"
    )
    schema[SCHEMA_KEYS.TASK_TYPE.value] = "MCQ"

    # Copying untransformed data fields
    schema[SCHEMA_KEYS.INPUT_INSTANCE.value] = f"Sentence: {test_instance['ctx']}\n"

    candidate_set, candidate_labels, key_dict = get_choices(test_instance["endings"])
    schema[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value] = candidate_set
    schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value] = candidate_labels

    schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_LABEL.value] = [
        key_dict[test_instance["label"]]
    ]
    schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value] = candidate_set[
        int(test_instance["label"])
    ]

    schema[SCHEMA_KEYS.TASK_PROMPT.value] = (
        "Given a sentence and "
        + str(len(schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value]))
        + " options: "
        + ", ".join(schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value])
        + " as candidate answers that complete the rest of the sentence, "
    )

    schema[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] = (
        schema[SCHEMA_KEYS.TASK_PROMPT.value]
        + ", select the value associated with the option label corresponding to the correct answer.\n"
    )

    schema[SCHEMA_KEYS.FINAL_SUFFIX_TASK_INSTRUCTION.value] = "\n"

    schema[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value] = schema[
        SCHEMA_KEYS.GROUND_TRUTH_ANSWER_LABEL.value
    ]

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


class Hellaswag:
    """
    This Class holds the Hellaswag dataset post transformation into the schema required
    """

    def __init__(self):
        """
        Initialize Hellaswag instance
        """
        super().__init__()
        # load the dataset
        self.dataset = load_dataset("Rowan/hellaswag", split="validation")

        # convert the dataset into the schema defined
        self.intermediate_representation = self.dataset.map(
            transform_hellaswag, remove_columns=self.dataset.column_names
        )
