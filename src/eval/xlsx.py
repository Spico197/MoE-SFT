import argparse
import statistics
from pathlib import Path

import pandas as pd

from src.utils.io import load_json


# fmt: off
TASKS = [
    {"name": "mmlu_abstract_algebra,mmlu_anatomy,mmlu_astronomy,mmlu_business_ethics,mmlu_clinical_knowledge,mmlu_college_biology,mmlu_college_chemistry,mmlu_college_computer_science,mmlu_college_mathematics,mmlu_college_medicine,mmlu_college_physics,mmlu_computer_security,mmlu_conceptual_physics,mmlu_econometrics,mmlu_electrical_engineering,mmlu_elementary_mathematics,mmlu_formal_logic,mmlu_global_facts,mmlu_high_school_biology,mmlu_high_school_chemistry,mmlu_high_school_computer_science,mmlu_high_school_european_history,mmlu_high_school_geography,mmlu_high_school_government_and_politics,mmlu_high_school_macroeconomics,mmlu_high_school_mathematics,mmlu_high_school_microeconomics,mmlu_high_school_physics,mmlu_high_school_psychology,mmlu_high_school_statistics,mmlu_high_school_us_history,mmlu_high_school_world_history,mmlu_human_aging,mmlu_human_sexuality,mmlu_international_law,mmlu_jurisprudence,mmlu_logical_fallacies,mmlu_machine_learning,mmlu_management,mmlu_marketing,mmlu_medical_genetics,mmlu_miscellaneous,mmlu_moral_disputes,mmlu_moral_scenarios,mmlu_nutrition,mmlu_philosophy,mmlu_prehistory,mmlu_professional_accounting,mmlu_professional_law,mmlu_professional_medicine,mmlu_professional_psychology,mmlu_public_relations,mmlu_security_studies,mmlu_sociology,mmlu_us_foreign_policy,mmlu_virology,mmlu_world_religions", "metric": "acc,none", "filename": "mmlu.json", "alias": "mmlu"},
    {"name": "bbh_fewshot_boolean_expressions,bbh_fewshot_causal_judgement,bbh_fewshot_date_understanding,bbh_fewshot_disambiguation_qa,bbh_fewshot_dyck_languages,bbh_fewshot_formal_fallacies,bbh_fewshot_geometric_shapes,bbh_fewshot_hyperbaton,bbh_fewshot_logical_deduction_five_objects,bbh_fewshot_logical_deduction_seven_objects,bbh_fewshot_logical_deduction_three_objects,bbh_fewshot_movie_recommendation,bbh_fewshot_multistep_arithmetic_two,bbh_fewshot_navigate,bbh_fewshot_object_counting,bbh_fewshot_penguins_in_a_table,bbh_fewshot_reasoning_about_colored_objects,bbh_fewshot_ruin_names,bbh_fewshot_salient_translation_error_detection,bbh_fewshot_snarks,bbh_fewshot_sports_understanding,bbh_fewshot_temporal_sequences,bbh_fewshot_tracking_shuffled_objects_five_objects,bbh_fewshot_tracking_shuffled_objects_seven_objects,bbh_fewshot_tracking_shuffled_objects_three_objects,bbh_fewshot_web_of_lies,bbh_fewshot_word_sorting", "metric": "exact_match,none", "filename": "bbh.json", "alias": "bbh"},
    {"name": "gsm8k_cot", "metric": "exact_match,get-answer", "filename": "reasoning.json"},
    {"name": "mbpp", "metric": "pass@1", "filename": "mbpp.json", "nesting": False},
    {"name": "boolq", "metric": "acc,none", "filename": "qa.json"},
    {"name": "arc_easy", "metric": "acc_norm,none", "filename": "qa.json"},
    {"name": "arc_challenge", "metric": "acc_norm,none", "filename": "qa.json"},
    {"name": "sciq", "metric": "acc_norm,none", "filename": "extend.json"},
    {"name": "piqa", "metric": "acc_norm,none", "filename": "extend.json"},
    {"name": "winogrande", "metric": "acc,none", "filename": "extend.json"},
    {"name": "asdiv", "metric": "acc,none", "filename": "extend.json"},
    {"name": "lambada_openai", "metric": "acc,none", "filename": "extend.json"},
    {"name": "openbookqa", "metric": "acc_norm,none", "filename": "extend.json"},
]
# fmt: on


def get_task_result(name: str, metric: str, filename: str, nesting: bool = True, alias: str = None):
    tasks = name.split(",")
    res = load_json(filename)
    task_to_res = res["results"] if nesting else res
    scores = [task_to_res[task][metric] for task in tasks]
    task_name = alias if alias else name
    return task_name, statistics.mean(scores) * 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()

    results_folder = Path(args.folder)
    results = {}
    for task in TASKS:
        task["filename"] = results_folder / task["filename"]
        task_name, val = get_task_result(**task)
        results[task_name] = [val]

    # convert results to excel
    df = pd.DataFrame(results)
    df.to_excel(results_folder / "res.xlsx")
