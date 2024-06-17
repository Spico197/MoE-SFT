import json
from pathlib import Path

from tqdm import tqdm
from loguru import logger
from tabulate import tabulate

from src.utils.io import load_json


TASK_MAP = {
    "mmlu": "mmlu_computer_security,mmlu_high_school_chemistry,mmlu_philosophy,mmlu_elementary_mathematics,mmlu_prehistory,mmlu_formal_logic,mmlu_high_school_mathematics,mmlu_econometrics,mmlu_moral_scenarios,mmlu_college_mathematics,mmlu_high_school_government_and_politics,mmlu_us_foreign_policy,mmlu_high_school_world_history,mmlu_conceptual_physics,mmlu_college_medicine,mmlu_international_law,mmlu_abstract_algebra,mmlu_logical_fallacies,mmlu_machine_learning,mmlu_medical_genetics,mmlu_public_relations,mmlu_college_biology,mmlu_marketing,mmlu_electrical_engineering,mmlu_anatomy,mmlu_high_school_us_history,mmlu_high_school_biology,mmlu_miscellaneous,mmlu_high_school_psychology,mmlu_sociology,mmlu_business_ethics,mmlu_high_school_geography,mmlu_human_aging,mmlu_high_school_statistics,mmlu_moral_disputes,mmlu_professional_psychology,mmlu_global_facts,mmlu_college_physics,mmlu_nutrition,mmlu_high_school_macroeconomics,mmlu_world_religions,mmlu_professional_medicine,mmlu_high_school_computer_science,mmlu_college_chemistry,mmlu_human_sexuality,mmlu_high_school_microeconomics,mmlu_astronomy,mmlu_professional_accounting,mmlu_high_school_european_history,mmlu_jurisprudence,mmlu_professional_law,mmlu_high_school_physics,mmlu_virology,mmlu_management,mmlu_college_computer_science,mmlu_clinical_knowledge,mmlu_security_studies",
    "bbh": "bbh_fewshot_boolean_expressions,bbh_fewshot_causal_judgement,bbh_fewshot_date_understanding,bbh_fewshot_disambiguation_qa,bbh_fewshot_dyck_languages,bbh_fewshot_formal_fallacies,bbh_fewshot_geometric_shapes,bbh_fewshot_hyperbaton,bbh_fewshot_logical_deduction_five_objects,bbh_fewshot_logical_deduction_seven_objects,bbh_fewshot_logical_deduction_three_objects,bbh_fewshot_movie_recommendation,bbh_fewshot_multistep_arithmetic_two,bbh_fewshot_navigate,bbh_fewshot_object_counting,bbh_fewshot_penguins_in_a_table,bbh_fewshot_reasoning_about_colored_objects,bbh_fewshot_ruin_names,bbh_fewshot_salient_translation_error_detection,bbh_fewshot_snarks,bbh_fewshot_sports_understanding,bbh_fewshot_temporal_sequences,bbh_fewshot_tracking_shuffled_objects_five_objects,bbh_fewshot_tracking_shuffled_objects_seven_objects,bbh_fewshot_tracking_shuffled_objects_three_objects,bbh_fewshot_web_of_lies,bbh_fewshot_word_sorting",
    "reasoning": "gsm8k_cot",
    "qa": "arc_easy,arc_challenge,boolq",
}

CODE_TASK_MAP = {
    "mbpp": "mbpp",
    # "humaneval": "humaneval",
}


def collect_results(result_dir: str, verbose: bool = True) -> dict:
    def _val(ins: dict, task: str) -> float:
        if task == "mmlu":
            return ins["acc,none"]
        elif task == "bbh":
            return ins["exact_match,none"]
        elif task == "reasoning":
            if ins["alias"] == "gsm8k_cot":
                return ins["exact_match,get-answer"]
            else:
                raise ValueError(f"task: {task}, ins: {ins}")
        elif task == "qa":
            if ins["alias"] in ["arc_easy", "arc_challenge"]:
                return ins["acc_norm,none"]
            elif ins["alias"] == "boolq":
                return ins["acc,none"]
            else:
                raise ValueError(f"task: {task}, ins: {ins}")
        elif task in ["mbpp", "humaneval"]:
            return ins["pass@1"] if ins is not None else -100
        else:
            raise ValueError(f"task {task} not supported, ins: {ins}")

    headers = []
    table_vals = []
    if verbose:
        logger.info(f"results from: {result_dir}")
    folder_p = Path(result_dir)
    tot_vals = []
    for name, tasks in tqdm(TASK_MAP.items(), desc="Collecting"):
        res_file = folder_p / f"{name}.json"
        if res_file.exists():
            try:
                res = load_json(res_file)
                res_candidates = []
                for task in tasks.split(","):
                    if task in res["results"]:
                        res_candidates.append(res["results"][task])
                vals = []
                for item in res_candidates:
                    vals.append(_val(item, name))
            except json.decoder.JSONDecodeError:
                vals = [-1.0]
        else:
            vals = [-1.0]
        if len(vals) == 0:
            avg = 0.0
        else:
            avg = sum(vals) / len(vals)
        tot_vals.append(avg)
        if verbose:
            logger.info(f"task: {name}, num: {len(tasks.split(','))}, avg: {100 * avg:.3f} %")
        headers.append(name)
        table_vals.append(f"{100 * avg:.2f}")

    code_vals = []
    for name, filename in CODE_TASK_MAP.items():
        res_file = folder_p / f"{filename}.json"
        if res_file.exists():
            res = load_json(res_file)
            val = _val(res[name], name)
        else:
            val = -100.0
            logger.warning(f"code: missing {name} in {str(res_file)}")
        code_vals.append(val)
        if verbose:
            logger.info(f"task: {name}, avg: {100 * val:.3f} %")

    if len(code_vals) == 0:
        code_avg = 0.0
    else:
        code_avg = sum(code_vals) / len(code_vals)
    headers.append("Code")
    tot_vals.append(code_avg)
    table_vals.append(f"{100 * code_avg:.2f}")
    if verbose:
        logger.info(f"code avg: {100 * code_avg:.3f} %")

    if len(tot_vals) == 0:
        tot_avg = 0.0
    else:
        tot_avg = sum(tot_vals) / len(tot_vals)
    headers.append("Average")
    table_vals.append(f"{100 * tot_avg:.2f}")
    logger.info(f"total avg: {100 * tot_avg:.3f} %")
    print(tabulate([table_vals], headers=headers))


if __name__ == "__main__":
    # collect_results("results/LLaMA-MoE-v1-3_5B-2_8-new")
    # collect_results("results/Sheared-LLaMA-2.7B")
    import sys
    res_folder = sys.argv[1]
    collect_results(res_folder)
