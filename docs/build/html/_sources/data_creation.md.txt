## Intermediate Schema Creation

This readme contains instructions on converting any dataset into intermediate schema. To begin with we will define the schema

```json
INTERMEDIATE_SCHEMA={
    "dataset": "",
    "hf_dataset": "",
    "task_type": "Generative/MCQ",
    "instruction_id": "",
    "dataset_input": "",
    "candidate_answer_set": [],
    "candidate_answer_label_space": [],
    "ground_truth_answer_label": "", 
    "ground_truth_answer_text": "",
    "dataset_instruction": "",
    "final_suffix_task_instruction": "",
    "final_prefix_task_instruction": "",
    "task_instructions": [],
    "instruction_output": [],
    "instruction_following_errors_set": [],
    "reasoning_error_set": [],
    "cot_instruction": [],
}
```

Here,
- instruction_id: Instruction Name, should match the file and function name
- dataset_input: Instance given to the LLM without any instruction. 
- candidate_answer_set: the list of all posssible answers for that instance
- candidate_answer_label_space: the list of all posssible answer labels
- dataset_instruction: Task Prompt. Task prompt should not define how to generate the answer.
- final_suffix_task_instruction: The final task instruction which gets appended to the input and original_task_instruction
- final_prefix_task_instruction: The final task instruction which gets prepended to the input and original_task_instruction
- instruction_following_errors_set: overridden by the specific instruction which gets applied
- reasoning_error_set": overridden by the specific instruction which gets applied
- cot_instruction":# overridden by the specific instruction which gets applied

Every instance needs to be converted to the above format. For example, an instance in BoolQ dataset looks like this

```json
{
    "passage": "Persian (/ˈpɜːrʒən, -ʃən/), also known by its endonym Farsi (فارسی fārsi (fɒːɾˈsiː) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet.",
    "question": "do iran and afghanistan speak the same language",
    "answer": true
}
```

will look like

```json
{
  "dataset": "BoolQ",
  "hf_dataset": "https://huggingface.co/datasets/google/boolq/",
  "task_type": "MCQ",
  "instruction_id": "",
  "instance_id": "",
  "dataset_input": "Passage: Once all the players have completed their hands, it is the dealer's turn. The dealer hand will not be completed if all players have either busted or received Blackjacks. The dealer then reveals the hidden card and must hit until the cards total 17 or more points. (At most tables the dealer also hits on a ``soft'' 17, i.e. a hand containing an ace and one or more other cards totaling six.) Players win by not busting and having a total higher than the dealer, or not busting and having the dealer bust, or getting a blackjack without the dealer getting a blackjack. If the player and dealer have the same total (not counting blackjacks), this is called a ``push'', and the player typically does not win or lose money on that hand. Otherwise, the dealer wins.\nQuestion: does the dealer have to hit on 16\nOptions: ",
  "candidate_answer_set": [],
  "candidate_answer_label_space": [],
  "ground_truth_answer_label": "A",
  "ground_truth_answer_text": "True",
  "dataset_instruction": "Given a passage and a boolean question, and the possible answer candidates 'A' or 'B', ",
  "final_prefix_task_instruction": "",
  "final_suffix_task_instruction": "",
  "task_instructions": [
    "Given a passage and a boolean question, and the possible answer candidates 'A' or 'B', ",
  ],
  "instruction_output": [
    "True",
  ],
  "instruction_following_errors_set": [],
  "reasoning_error_set": [],
  "cot_instruction": "",
  "CLASSIFICATION": "",
  "input": "",
  "COT": "True"
}
```

## Adding a new dataset
To add a new dataset, create a new class file. The class file should have the same name as the class name for that dataset. For example, to add `BoolQ` dataset, the filename should be `BoolQ.py` and the file should have a class named `BoolQ`. The class should load the dataset and convert each instance to the schema prescribed above. The file `BoolQ.py` can be used as reference for creating such a file.