from pathlib import Path

import pandas as pd


class TABLE_HEADERS:
    headers = [
        "instr_strict_em",
        "pca_strict_em",
        "instr_loose_em",
        "pca_loose_em",
        "pca_label_em",
        "instr_count",
        "pca_count",
        "pca_label_count",
        "instr_reason_error",
        "pca_reason_error",
        "pca_label_reason_error",
        "instr_if_error",
        "pca_if_error",
        "pca_label_if_error",
        "instr_reason_count",
        "pca_reason_count",
        "pca_label_reason_count",
        "instr_if_count",
        "pca_if_count",
        "pca_label_if_count",
        "instr_unclass_count",
        "pca_unclass_count",
        "pca_label_unclass_count",
        "instr_loose_reason",
        "instr_loose_if",
        "instr_loose_unclass",
        "instr_strict_em_count",
        "instr_loose_em_count",
    ]

    def __init__(self):
        pass


class TABLE_ORDERING:
    categories = [
        "print_correct_answer_label",
        "print_correct_answer",
        "String Manipulation",
        "Numeric Manipulation",
        "Format Correct Answer",
        "Operations on List (Conditional)",
        "Operations on List",
        "Label Manipulation",
    ]

    models = [
        "granite-8b-code-instruct-4k",
        "gemma-2-9b",
        "gemma-2-27b-it",
        "Phi-3-medium-4k-instruct",
        "Qwen2.5-14B-Instruct",
        "Qwen2.5-32B-Instruct",
        "Qwen2.5-72B-Instruct",
        "Meta-Llama-3.1-70B-Instruct",
    ]

    def __init__(self):
        pass


class DATASETS:
    datasets = ["MMLUPro", "Piqa", "Winogrande", "BBH", "BoolQ", "MathQA"]

    def __init__(self):
        pass


class CATEGORY_MAPPING:
    mapping = {
        "alternate_case_correct_answer": "String Manipulation",
        "capitalize_correct_answer": "String Manipulation",
        "reverse_correct_answer_alternate_case": "String Manipulation",
        "reverse_correct_answer": "String Manipulation",
        "flip_binary_classification_labels": "Label Manipulation",
        "flip_binary_classification_text": "Label Manipulation",
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
        "Label Manipulation",
        "Format Correct Answer",
        "No Manipulation",
        "Numeric Manipulation",
        "Operations on List",
        "Operations on List (Conditional)",
    ]

    def __init__(self):
        pass


class BaseMetric:
    def __init__(self):
        self.num_instances = 0
        # Exact Match
        self.em_strict = 0
        self.em_loose = 0
        self.em_pca_strict = 0
        self.em_pca_loose = 0
        self.em_pca_count = 0
        self.em_pca_label_strict = 0
        self.em_pca_label_loose = 0
        self.em_pca_label_count = 0
        # Reasoning, Instr. follow, Unclassified
        self.reason_strict = 0
        self.reason_count = 0
        self.reason_loose = 0
        self.if_strict = 0
        self.if_count = 0
        self.if_loose = 0
        self.unclass_strict = 0
        self.unclass_count = 0
        self.unclass_loose = 0
        # PCA versions
        self.pca_reason = 0
        self.pca_reason_count = 0
        self.pca_if = 0
        self.pca_if_count = 0
        self.pca_unclass = 0
        self.pca_unclass_count = 0
        self.pca_label_reason = 0
        self.pca_label_reason_count = 0
        self.pca_label_if = 0
        self.pca_label_if_count = 0
        self.pca_label_unclass = 0
        self.pca_label_unclass_count = 0

    def update_metrics(self, em_metrics: dict, analysis_metrics: dict):
        self.num_instances += 1
        self.em_strict += em_metrics["strict"]
        self.em_loose += em_metrics["loose"]
        self.em_pca_strict += em_metrics["pca_strict"]
        self.em_pca_loose += em_metrics["pca_loose"]
        self.em_pca_count += em_metrics["pca_count"]
        self.em_pca_label_strict += em_metrics["pca_label_strict"]
        self.em_pca_label_loose += em_metrics["pca_label_loose"]
        self.em_pca_label_count += em_metrics["pca_label_count"]

        self.reason_strict += analysis_metrics["reason_strict"]
        self.reason_loose += analysis_metrics["reason_loose"]
        self.reason_count += analysis_metrics["reason_loose"]
        self.if_strict += analysis_metrics["if_strict"]
        self.if_loose += analysis_metrics["if_loose"]
        self.if_count += analysis_metrics["if_loose"]
        self.unclass_strict += analysis_metrics["unclass_strict"]
        self.unclass_loose += analysis_metrics["unclass_loose"]
        self.unclass_count += analysis_metrics["unclass_loose"]
        self.pca_reason += analysis_metrics["pca_reason"]
        self.pca_reason_count += analysis_metrics["pca_reason"]
        self.pca_if += analysis_metrics["pca_if"]
        self.pca_if_count += analysis_metrics["pca_if"]
        self.pca_unclass += analysis_metrics["pca_unclass"]
        self.pca_unclass_count += analysis_metrics["pca_unclass"]
        self.pca_label_reason += analysis_metrics["pca_label_reason"]
        self.pca_label_reason_count += analysis_metrics["pca_label_reason"]
        self.pca_label_if += analysis_metrics["pca_label_if"]
        self.pca_label_if_count += analysis_metrics["pca_label_if"]
        self.pca_label_unclass += analysis_metrics["pca_label_unclass"]
        self.pca_label_unclass_count += analysis_metrics["pca_label_unclass"]

    def return_metrics(self):
        if self.em_pca_count > 0:
            pca_strict_em = self.em_pca_strict / self.em_pca_count
            pca_loose_em = self.em_pca_loose / self.em_pca_count
            pca_reason = self.pca_reason / self.em_pca_count
            pca_if = self.pca_if / self.em_pca_count
        else:
            pca_strict_em = 0
            pca_loose_em = 0
            pca_reason = 0
            pca_if = 0
        if self.em_pca_label_count > 0:
            pca_label_strict_em = self.em_pca_label_strict / self.em_pca_label_count
            pca_label_loose_em = self.em_pca_label_loose / self.em_pca_label_count
            pca_label_reason = self.pca_label_reason / self.em_pca_label_count
            pca_label_if = self.pca_label_if / self.em_pca_label_count
        else:
            pca_label_strict_em = 0
            pca_label_loose_em = 0
            pca_label_reason = 0
            pca_label_if = 0

        return [
            self.em_strict / self.num_instances,
            pca_strict_em,
            self.em_loose / self.num_instances,
            pca_loose_em,
            pca_label_loose_em,
            self.num_instances,
            self.em_pca_count,
            self.em_pca_label_count,
            self.reason_loose / self.num_instances,
            pca_reason,
            pca_label_reason,
            self.if_loose / self.num_instances,
            pca_if,
            pca_label_if,
            self.reason_count,
            self.pca_reason_count,
            self.pca_label_reason_count,
            self.if_count,
            self.pca_if_count,
            self.pca_label_if_count,
            self.unclass_count,
            self.pca_unclass_count,
            self.pca_label_unclass_count,
            self.reason_loose / self.num_instances,
            self.if_loose / self.num_instances,
            self.unclass_loose / self.num_instances,
            self.em_strict,
            self.em_loose,
        ]


class DatasetMetric:
    def __init__(self, data):
        self.metrics = {}
        unique_datasets = [tst["dataset"] for tst in data]
        unique_datasets = list(set(unique_datasets))
        for dataset in unique_datasets:
            self.metrics[dataset] = BaseMetric()

    def update_metrics(self, dataset: str, em_metrics: dict, analysis_metrics: dict):
        self.metrics[dataset].update_metrics(em_metrics, analysis_metrics)

    def return_metrics(self):
        table_metrics = []
        for dset in list(self.metrics.keys()):
            metric = self.metrics[dset].return_metrics()
            metric.insert(0, dset)
            table_metrics.append(metric)
        return table_metrics


class InstructionMetric:
    def __init__(self, data):
        self.metrics = {}
        unique_instructions = [tst["instruction_id"] for tst in data]
        unique_instructions = list(set(unique_instructions))
        for instruction in unique_instructions:
            self.metrics[instruction] = BaseMetric()

    def update_metrics(
        self, instruction: str, em_metrics: dict, analysis_metrics: dict
    ):
        self.metrics[instruction].update_metrics(em_metrics, analysis_metrics)

    def return_metrics(self):
        table_metrics = []
        for instr in list(self.metrics.keys()):
            metric = self.metrics[instr].return_metrics()
            metric.insert(0, instr)
            table_metrics.append(metric)
        return table_metrics


class ClassificationMetric:
    def __init__(self, data):
        self.metrics = {}
        category_map = CATEGORY_MAPPING()
        unique_classes = [
            category_map.mapping[tst["instruction_id"]]
            for tst in data
            if tst["instruction_id"]
            not in ["print_correct_answer_label", "print_correct_answer"]
        ]
        unique_classes = list(set(unique_classes))
        if any(tst["instruction_id"] == "print_correct_answer_label" for tst in data):
            unique_classes.append("print_correct_answer_label")
        if any(tst["instruction_id"] == "print_correct_answer" for tst in data):
            unique_classes.append("print_correct_answer")

        for classification in unique_classes:
            self.metrics[classification] = BaseMetric()

    def update_metrics(
        self,
        classification: str,
        instruction: str,
        em_metrics: dict,
        analysis_metrics: dict,
    ):
        if classification in self.metrics.keys() or instruction in self.metrics.keys():
            key = (
                instruction
                if instruction in ["print_correct_answer", "print_correct_answer_label"]
                else classification
            )
            self.metrics[key].update_metrics(em_metrics, analysis_metrics)

    def return_metrics(self):
        table_metrics = []
        for cat in list(self.metrics.keys()):
            metric = self.metrics[cat].return_metrics()
            metric.insert(0, cat)
            table_metrics.append(metric)
        return table_metrics


class DatasetInstrMetric:
    def __init__(self, data):
        self.metrics = {}
        category_map = CATEGORY_MAPPING().mapping
        unique_datasets = [tst["dataset"] for tst in data]
        unique_datasets = list(set(unique_datasets))
        for dataset in unique_datasets:
            subset = list(filter(lambda d: d["dataset"] == dataset, data))
            unique_instructions = [tst["instruction_id"] for tst in subset]
            unique_instructions = list(set(unique_instructions))
            if len(unique_instructions) > 0:
                self.metrics[dataset] = {}
            for instruction in unique_instructions:
                if (
                    dataset in ["Piqa", "BoolQ", "Winogrande"]
                    and category_map[instruction] == "Operations on List"
                ):
                    continue
                self.metrics[dataset][instruction] = BaseMetric()

    def update_metrics(self, dataset, instruction, em_metrics, analysis_metrics):
        if (
            dataset in self.metrics.keys()
            and instruction in self.metrics[dataset].keys()
        ):
            self.metrics[dataset][instruction].update_metrics(
                em_metrics, analysis_metrics
            )

    def return_metrics(self):
        table_metrics = []
        for dataset in list(self.metrics.keys()):
            for instruction in list(self.metrics[dataset].keys()):
                metric = self.metrics[dataset][instruction].return_metrics()
                metric.insert(0, instruction)
                metric.insert(0, dataset)
                table_metrics.append(metric)
        return table_metrics


class ClassificationInstrMetric:
    def __init__(self, data):
        self.metrics = {}
        category_map = CATEGORY_MAPPING()
        unique_classes = [category_map.mapping[tst["instruction_id"]] for tst in data]
        unique_classes = list(set(unique_classes))

        for classification in unique_classes:
            subset = list(
                filter(
                    lambda d: category_map.mapping[d["instruction_id"]]
                    == classification,
                    data,
                )
            )
            unique_instructions = [tst["instruction_id"] for tst in subset]
            unique_instructions = list(set(unique_instructions))
            if len(unique_instructions) > 0:
                self.metrics[classification] = {}
            for instruction in unique_instructions:
                self.metrics[classification][instruction] = BaseMetric()

    def update_metrics(self, classification, instruction, em_metrics, analysis_metrics):
        if (
            classification in self.metrics.keys()
            and instruction in self.metrics[classification].keys()
        ):
            self.metrics[classification][instruction].update_metrics(
                em_metrics, analysis_metrics
            )

    def return_metrics(self):
        table_metrics = []
        for cat in list(self.metrics.keys()):
            for instruction in list(self.metrics[cat].keys()):
                metric = self.metrics[cat][instruction].return_metrics()
                metric.insert(0, instruction)
                metric.insert(0, cat)
                table_metrics.append(metric)
        return table_metrics


class ClassificationDatasetMetric:
    def __init__(self, data):
        self.metrics = {}
        category_map = CATEGORY_MAPPING()
        datasets = DATASETS().datasets

        unique_classes = [
            category_map.mapping[tst["instruction_id"]]
            for tst in data
            if tst["instruction_id"]
            not in ["print_correct_answer_label", "print_correct_answer"]
        ]
        unique_classes = list(set(unique_classes))
        if any(tst["instruction_id"] == "print_correct_answer_label" for tst in data):
            unique_classes.append("print_correct_answer_label")
        if any(tst["instruction_id"] == "print_correct_answer" for tst in data):
            unique_classes.append("print_correct_answer")

        for classification in unique_classes:
            if classification == "print_correct_answer_label":
                subset = list(
                    filter(lambda d: d["instruction_id"] == classification, data)
                )
            elif classification == "print_correct_answer":
                subset = list(
                    filter(lambda d: d["instruction_id"] == classification, data)
                )
            else:
                subset = list(
                    filter(
                        lambda d: category_map.mapping[d["instruction_id"]]
                        == classification,
                        data,
                    )
                )
            unique_dataset = [tst["dataset"] for tst in subset]
            parent_level_unique_dataset = []
            for dset in unique_dataset:
                for dataset in datasets:
                    if dataset in dset:
                        parent_level_unique_dataset.append(dataset)
                        break
            unique_dataset = list(set(parent_level_unique_dataset))

            if len(unique_dataset) > 0:
                self.metrics[classification] = {}
            for dataset in unique_dataset:
                if classification == "Operations on List" and dataset in [
                    "Piqa",
                    "Winogrande",
                    "BoolQ",
                ]:
                    continue
                self.metrics[classification][dataset] = BaseMetric()

    def update_metrics(
        self, classification, parent_dataset, instruction, em_metrics, analysis_metrics
    ):
        if classification in self.metrics.keys() or instruction in self.metrics.keys():
            key = (
                instruction
                if instruction in ["print_correct_answer", "print_correct_answer_label"]
                else classification
            )
            if parent_dataset in self.metrics[key].keys():
                self.metrics[key][parent_dataset].update_metrics(
                    em_metrics, analysis_metrics
                )

    def return_metrics(self):
        table_metrics = []
        for cat in list(self.metrics.keys()):
            for dset in list(self.metrics[cat].keys()):
                metric = self.metrics[cat][dset].return_metrics()
                metric.insert(0, dset)
                metric.insert(0, cat)
                table_metrics.append(metric)
        return table_metrics


class ClassificationDatasetNoIFMetric:
    def __init__(self, data):
        self.metrics = {}
        category_map = CATEGORY_MAPPING()
        datasets = DATASETS().datasets

        unique_classes = [
            category_map.mapping[tst["instruction_id"]]
            for tst in data
            if tst["instruction_id"]
            not in ["print_correct_answer_label", "print_correct_answer"]
        ]
        unique_classes = list(set(unique_classes))
        if any(tst["instruction_id"] == "print_correct_answer_label" for tst in data):
            unique_classes.append("print_correct_answer_label")
        if any(tst["instruction_id"] == "print_correct_answer" for tst in data):
            unique_classes.append("print_correct_answer")

        for classification in unique_classes:
            if classification == "print_correct_answer":
                subset = list(
                    filter(lambda d: d["instruction_id"] == classification, data)
                )
            elif classification == "print_correct_answer_label":
                subset = list(
                    filter(lambda d: d["instruction_id"] == classification, data)
                )
            else:
                subset = list(
                    filter(
                        lambda d: category_map.mapping[d["instruction_id"]]
                        == classification,
                        data,
                    )
                )
            unique_dataset = [tst["dataset"] for tst in subset]
            parent_level_unique_dataset = []
            for dset in unique_dataset:
                for dataset in datasets:
                    if dataset in dset:
                        parent_level_unique_dataset.append(dataset)
                        break
            unique_dataset = list(set(parent_level_unique_dataset))

            if len(unique_dataset) > 0:
                self.metrics[classification] = {}
            for dataset in unique_dataset:
                if classification == "Operations on List" and dataset in [
                    "Piqa",
                    "Winogrande",
                    "BoolQ",
                ]:
                    continue
                self.metrics[classification][dataset] = BaseMetric()

    def update_metrics(
        self, classification, parent_dataset, instruction, em_metrics, analysis_metrics
    ):
        if classification in self.metrics.keys() or instruction in self.metrics.keys():
            key = (
                instruction
                if instruction in ["print_correct_answer", "print_correct_answer_label"]
                else classification
            )
            if parent_dataset in self.metrics[key].keys():
                self.metrics[key][parent_dataset].update_metrics(
                    em_metrics, analysis_metrics
                )

    def return_metrics(self):
        table_metrics = []
        for cat in list(self.metrics.keys()):
            for dset in list(self.metrics[cat].keys()):
                metric = self.metrics[cat][dset].return_metrics()
                metric.insert(0, dset)
                metric.insert(0, cat)
                table_metrics.append(metric)
        return table_metrics


def write_result_xlsx(
    output_file,
    dataset_csv,
    instruction_csv,
    classification_csv,
    data_instr_csv,
    class_instr_csv,
    class_data_csv,
    class_data_noif_csv,
):
    table_headers = TABLE_HEADERS().headers
    category_table_ordering = TABLE_ORDERING().categories

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = pd.ExcelWriter(output_file, engine="xlsxwriter")
    dataset_headers = ["dataset"] + table_headers
    data_df = pd.DataFrame(dataset_csv, columns=dataset_headers)
    data_df.to_excel(writer, sheet_name="dataset", index=False)

    instruction_headers = ["instruction"] + table_headers
    instr_df = pd.DataFrame(instruction_csv, columns=instruction_headers)
    instr_df.to_excel(writer, sheet_name="instruction", index=False)

    classification_headers = ["classification"] + table_headers
    class_df = pd.DataFrame(classification_csv, columns=classification_headers)
    class_df["classification"] = pd.Categorical(
        class_df["classification"], ordered=True, categories=category_table_ordering
    )
    class_df = class_df.sort_values("classification")
    class_df.to_excel(writer, sheet_name="classification", index=False)

    data_instr_headers = ["dataset", "instruction"] + table_headers
    data_instr_df = pd.DataFrame(data_instr_csv, columns=data_instr_headers)
    data_instr_df.to_excel(writer, sheet_name="dataset_instructions", index=False)

    class_instr_headers = ["classification", "instruction"] + table_headers
    class_instr_df = pd.DataFrame(class_instr_csv, columns=class_instr_headers)
    class_instr_df["classification"] = pd.Categorical(
        class_instr_df["classification"],
        ordered=True,
        categories=category_table_ordering,
    )
    class_instr_df = class_instr_df.sort_values("classification")
    class_instr_df.to_excel(
        writer, sheet_name="classification_instructions", index=False
    )

    class_data_headers = ["classification", "dataset"] + table_headers
    class_data_df = pd.DataFrame(class_data_csv, columns=class_data_headers)
    class_data_df["classification"] = pd.Categorical(
        class_data_df["classification"],
        ordered=True,
        categories=category_table_ordering,
    )
    class_data_df = class_data_df.sort_values("classification")
    class_data_df.to_excel(writer, sheet_name="classification_dataset", index=False)

    class_data_noif_headers = ["classification", "dataset"] + table_headers
    class_data_noif_df = pd.DataFrame(
        class_data_noif_csv, columns=class_data_noif_headers
    )
    class_data_noif_df["classification"] = pd.Categorical(
        class_data_noif_df["classification"],
        ordered=True,
        categories=category_table_ordering,
    )
    class_data_noif_df = class_data_noif_df.sort_values("classification")
    class_data_noif_df.to_excel(
        writer, sheet_name="noif_classification_dataset", index=False
    )
    writer.close()
    return instr_df, class_df, class_data_df
