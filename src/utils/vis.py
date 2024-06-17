import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def heatmap(
    data: np.ndarray,
    title: str = None,
    xlabels: list = None,
    ylabels: list = None,
    filepath: str = None,
):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    if title:
        ax.set_title(title)
    ax.imshow(data, cmap=mpl.colormaps["OrRd"], interpolation="nearest")
    shape = data.shape
    for row in range(shape[0]):
        for col in range(shape[1]):
            ax.text(
                col,
                row,
                f"{data[row, col]:.2f}",
                ha="center",
                va="center",
                color="black",
            )
    if xlabels:
        ax.set_xticks(np.arange(len(xlabels)), xlabels)
    if ylabels:
        ax.set_yticks(np.arange(len(ylabels)), ylabels)
    if filepath:
        fig.savefig(filepath)
    plt.close()


def group_bar(name2data: dict, labels: list, title: str = None, filepath: str = None):
    x = np.arange(len(labels))
    width = 0.8 / len(name2data)
    multiplier = 0
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in name2data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, data, width, label=name)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.legend()
    ax.set_xticks(x + width, labels)
    if title:
        ax.set_title(title)
    if filepath:
        fig.savefig(filepath)
    plt.close()


def bar(name2data: dict, title: str = None, filepath: str = None):
    names = sorted(name2data.keys())
    data = [name2data[name] for name in names]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(names, data)
    if title:
        ax.set_title(title)
    ax.legend()
    if filepath:
        fig.savefig(filepath)
    plt.close()


def gate_load_stats(model_dir, data_dir, result_dir, update_strategy: str = "cos"):
    from pathlib import Path

    import torch
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from tqdm import tqdm, trange

    from src.core.train import get_model_and_tokenizer, fault_tolerance_data_collator
    from src.data import get_cached_datasets_from_dir
    from src.callbacks import AdaptiveSamplingCallback

    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True, parents=True)
    ac = Accelerator()
    model, tokenizer = get_model_and_tokenizer(
        "llama_moe",
        model_dir,
        trust_remote_code=True,
        padding_side="right",
        torch_dtype=torch.bfloat16,
        attn_impl="flash_attention_2",
        model_max_length=2048,
        cache_dir=None,
    )
    model.eval()
    model = ac.prepare_model(model)
    eval_dataset_map = get_cached_datasets_from_dir(data_dir, tokenizer)

    names = sorted(eval_dataset_map.keys())
    all_gate_load_list = []
    for name in names:
        eval_dataset = eval_dataset_map[name]
        gate_load_list = []
        loader = DataLoader(
            eval_dataset, batch_size=4, collate_fn=fault_tolerance_data_collator
        )
        loader = ac.prepare_data_loader(loader)

        # from src.utils.debugging import remote_breakpoint
        # remote_breakpoint()

        layer2expert2tokens = {}
        for l in range(32):
            layer2expert2tokens[l] = {}
            for e in range(8):
                layer2expert2tokens[l][e] = []
        with torch.inference_mode():
            for batch_idx, batch in enumerate(tqdm(loader, desc=name)):
                if batch_idx >= 20:
                    break
                outs = model(**batch, output_attentions=False, use_cache=False)
                # gate_load: (tensor([1.0, 2.3, ... num_experts]), tensor([3.0, 4.5, ... num_experts]), ... num_layers)
                gate_load = outs.gate_load
                # (num_layers, num_experts)
                gate_load = torch.stack(gate_load, dim=0).detach().cpu().numpy()
                gate_load_list.append(gate_load)
                for l, expert2tokens in enumerate(outs.expert2tokens):
                    tokens = tokenizer.convert_ids_to_tokens(
                        batch["input_ids"].flatten().tolist()
                    )
                    for e, _tokens in expert2tokens.items():
                        selected_tokens = [tokens[_t] for _t in _tokens]
                        layer2expert2tokens[l][e].extend(selected_tokens)
        for e, tokens in layer2expert2tokens[31].items():
            print(
                f"name: {name}, expert: {e}, selected: {len(tokens)}, set_of_selected: {len(set(tokens))}"
            )
            print(f"set of tokens: {sorted(set(tokens))}")
            print(f"tokens: {sorted(tokens)}\n\n")
        # (num_batches, num_layers, num_experts)
        gate_load_arr = np.stack(gate_load_list, axis=0)
        # (num_layers, num_experts)
        gate_load_sum = gate_load_arr.sum(axis=0)
        all_gate_load_list.append(gate_load_sum)

    # (num_datasets, num_layers, num_experts)
    all_gate_load_arr = np.stack(all_gate_load_list, axis=0)
    # (num_layers, num_datasets, num_experts)
    all_gate_load_arr = all_gate_load_arr.transpose((1, 0, 2))
    all_gate_load_arr = all_gate_load_arr / all_gate_load_arr.sum(
        axis=-1, keepdims=True
    )
    np.save(result_dir / "gate_load.npy", all_gate_load_arr)
    for layer_idx in trange(all_gate_load_arr.shape[0], desc="Dumping"):
        loads = all_gate_load_arr[layer_idx]
        heatmap(
            loads,
            title=f"gate load of layer {layer_idx}",
            ylabels=names,
            filepath=result_dir / f"gate_load-L{layer_idx}.png",
        )
        if update_strategy == "cos":
            norm = np.linalg.norm(loads, axis=1, keepdims=True)
            normalized = loads / norm
            sim = np.dot(normalized, normalized.T)
            delta = 1.0 - sim
        elif update_strategy == "l2":
            sim = np.dot(loads, loads.T)
            delta = np.linalg.norm(loads[:, np.newaxis] - loads, axis=2)
        else:
            raise ValueError(f"Invalid update strategy: {update_strategy}")
        heatmap(
            sim,
            title=f"gate load dot similarity of layer {layer_idx}",
            xlabels=names,
            ylabels=names,
            filepath=result_dir / f"gate_load-dot_sim-L{layer_idx}.png",
        )
        heatmap(
            delta,
            title=f"gate load delta of layer {layer_idx}",
            xlabels=names,
            ylabels=names,
            filepath=result_dir / f"gate_load-delta-L{layer_idx}.png",
        )
        _delta_vec = delta.mean(axis=1)
        ori_weights = np.ones_like(_delta_vec)
        ori_weights /= ori_weights.sum()
        new_weights = AdaptiveSamplingCallback._update_weights(ori_weights, _delta_vec)
        old_new_plot_data = {}
        for name in names:
            name_idx = names.index(name)
            old_new_plot_data[name] = [ori_weights[name_idx], new_weights[name_idx]]
        group_bar(
            old_new_plot_data,
            ["old", "new"],
            title=f"updated weights of layer {layer_idx}",
            filepath=result_dir / f"gate_load-prob-L{layer_idx}.png",
        )


def sampling_info_stats(filepath: str, output_dir: str):
    from pathlib import Path
    from collections import defaultdict
    import numpy as np
    from src.utils.io import load_jsonlines

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    data = load_jsonlines(filepath)
    step2data = {ins["step"]: ins for ins in data}

    data_types = sorted(data[0]["old_prob_map"].keys())
    xtick_labels = ["Code", "Math", "OpenOrca", "ShareGPT"]
    steps = sorted(step2data.keys())

    probs = defaultdict(list)
    loads = defaultdict(list)
    sims = defaultdict(list)
    for step in steps:
        ins = step2data[step]
        for data_type in data_types:
            probs[data_type].append(ins["old_prob_map"][data_type])
            load = ins["name2load"][data_type]
            # load = np.array(load)
            # load = load / load.sum()
            # print(f"{data_type} load: {load.tolist()}")
            loads[data_type].append(load)
            # sims[data_type].append(ins["sim"][data_type])

    # probs
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    for data_type in data_types:
        ax.plot(steps, probs[data_type], label=data_type)
    ax.set_title("Dynamic Sampling Weights")
    ax.legend(loc="lower right")
    ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/prob.pdf")

    # loads
    def cv_square(data):
        return np.var(data) / (np.mean(data) ** 2 + 1e-10)

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    bar_width = 0.35
    for i, data_type in enumerate(data_types):
        first = cv_square(loads[data_type][0])
        first_pos = i - bar_width / 2
        last = cv_square(loads[data_type][-1])
        last_pos = i + bar_width / 2
        print(f"{data_type} load_cv: {first} -> {last}, delta: {last - first}")
        ax.bar(
            first_pos,
            first,
            bar_width,
            hatch="||",
            edgecolor="#6999d0",
            linewidth=2,
            facecolor="white",
        )
        ax.bar(
            last_pos,
            last,
            bar_width,
            hatch="//",
            edgecolor="#e08b4e",
            linewidth=2,
            facecolor="white",
        )
    ax.set_xticks([i for i in range(len(data_types))])  # Center ticks between bars
    ax.set_xticklabels(xtick_labels)
    ax.set_title("CV(load)^2")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(["Beginning", "End"], loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/load_cv.pdf")

    # # sims
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(steps, np.mean(sims, axis=1))
    # ax.set_title(f"Mean Similarities with {data_type}")
    # ax.set_xlabel("step")
    # fig.savefig(f"{output_dir}/sim.png")

    # distances
    for data_type in data_types:
        print(f"diff: {data_type}")
        load = np.array(loads[data_type])
        delta = np.linalg.norm(load[:, np.newaxis] - load, axis=2)
        mean_delta = np.mean(delta, axis=1)
        for i in range(len(load)):
            print(mean_delta[i])

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    bar_width = 0.35
    first_loads = np.array([loads[data_types[i]][0] for i in range(len(data_types))])
    first_delta = np.linalg.norm(first_loads[:, np.newaxis] - first_loads, axis=2)
    first_mean_delta = np.mean(first_delta, axis=1)
    last_loads = np.array([loads[data_types[i]][-1] for i in range(len(data_types))])
    last_delta = np.linalg.norm(last_loads[:, np.newaxis] - last_loads, axis=2)
    last_mean_delta = np.mean(last_delta, axis=1)
    for i, data_type in enumerate(data_types):
        first_delta = first_mean_delta[i]
        first_pos = i - bar_width / 2
        last_delta = last_mean_delta[i]
        last_pos = i + bar_width / 2
        print(
            f"{data_type} diff: {first_delta} -> {last_delta}, delta: {last_delta - first_delta}"
        )
        ax.bar(
            first_pos,
            first_delta,
            bar_width,
            hatch="||",
            edgecolor="#6999d0",
            linewidth=2,
            facecolor="white",
        )
        ax.bar(
            last_pos,
            last_delta,
            bar_width,
            hatch="//",
            edgecolor="#e08b4e",
            linewidth=2,
            facecolor="white",
        )
    ax.set_xticks([i for i in range(len(data_types))])  # Center ticks between bars
    ax.set_xticklabels(xtick_labels)
    ax.set_title("Expert Selection Distances (L2)")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(["Beginning", "End"], loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/diff.pdf")


def sampling_prob_stats(model_id: str, filepath: str, output_dir: str):
    from pathlib import Path
    from collections import defaultdict
    from src.utils.io import load_jsonlines

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    data = load_jsonlines(filepath)
    step2data = {ins["step"]: ins for ins in data}
    steps = sorted(step2data.keys())
    data_types = sorted(data[0]["old_prob_map"].keys())
    type2probs = defaultdict(list)
    for step in steps:
        ins = step2data[step]
        for data_type in data_types:
            type2probs[data_type].append(ins["old_prob_map"][data_type])
    print(f"#Steps: {len(steps)}")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for data_type in data_types:
        ax.plot(steps, type2probs[data_type], label=data_type)
    ax.set_xlim(0, 2000)
    ax.legend()
    ax.grid(True, zorder=0)
    ax.set_title("Sampling Probability")
    ax.set_xlabel("step")
    fig.savefig(f"{output_dir}/prob-{model_id}.png")


def test_sampling_convergence():
    from collections import defaultdict
    from src.callbacks import AdaptiveSamplingCallback, RandomSamplingCallback

    # dynamic: 10.0, 5e-2
    name2load = {"code": [0.14031716417910448, 0.1310634328358209, 0.12651119402985075, 0.10993470149253731, 0.12196828358208955, 0.12552238805970148, 0.12791977611940297, 0.11676305970149255], "orca": [0.15106234655836084, 0.11803640166095838, 0.12349968175067437, 0.12884551268450883, 0.11344072985178673, 0.1383778377231534, 0.11733170672566907, 0.1094057830448883], "math": [0.16001617686708006, 0.10756444371505268, 0.11391210568886491, 0.114803005615014, 0.11676650216277679, 0.1177863481308685, 0.13630182751708533, 0.13284959030325763], "sharegpt": [0.15440024978412215, 0.113654214863131, 0.12914741653941664, 0.12104040941178769, 0.11470799162832905, 0.13593110446537907, 0.12316259873058931, 0.10795601457724527]}  # fmt: skip
    # moduleformer: 10.0, 8e-1
    # name2load = {"code": [0.0009968102073365233, 0.0001812382195157315, 0.0009786863853849502, 0.0008518196317239381, 0.13205016673916198, 0.000724952878062926, 5.437146585471945e-05, 0.01346599971001885, 0.08014354066985646, 5.437146585471945e-05, 0.01904813687110338, 0.0, 0.0007793243439176454, 0.000724952878062926, 0.09194214876033059, 0.23983253588516748, 0.19155067420617664, 0.0, 0.07307525010874294, 0.023071625344352618, 0.0006343337683050603, 0.0178882122662027, 0.005274032187907787, 0.03858561693489924, 1.8123821951573152e-05, 0.00028998115122517043, 0.019120632158909672, 0.0003987240829346093, 0.0016855154414963031, 0.04594388864723793, 0.0005437146585471945, 9.061910975786575e-05], "orca": [0.0020665409020751155, 8.574858514834504e-06, 0.004073057794546389, 0.002057966043560281, 0.027208026067569883, 0.01708111816155033, 0.0006945635397015948, 0.0287772251757846, 0.043568856113874115, 0.028219859372320355, 0.004115932087120562, 0.0, 0.11049562682215741, 0.0468959012176299, 0.20437317784256556, 0.1614903104098782, 0.01911335962956611, 0.0, 0.015100325844623562, 0.009715314697307492, 0.0004544675012862287, 0.022629051620648252, 0.0011490310409878236, 0.0032927456696964495, 0.001020408163265306, 0.0021179900531641226, 0.010049734179386038, 0.009783913565426168, 0.01661807580174927, 0.19773623735208365, 0.0039358600583090375, 0.006156748413651173], "math": [0.03253549581906953, 7.014984005836466e-05, 0.002118525169762613, 0.0012065772490038721, 0.11881979909085806, 0.003928391043268421, 0.03210056681070767, 0.045218586901621866, 0.06749817610415848, 0.0010803075368988157, 0.01540490487681688, 0.0, 0.018126718671081427, 0.006930804197766429, 0.16879454514843703, 0.2359980919243504, 0.027821426567147426, 0.0, 0.0370110556147932, 0.017397160334474436, 0.0002525394242101128, 0.015909983725237106, 0.0015713564173073684, 0.02965935237667658, 0.00023850945619843984, 0.0005191088164318985, 0.005906616532914305, 0.0018379258095291542, 0.0071693136539648684, 0.10159099837252371, 0.002665693922217857, 0.0006173185925136091], "sharegpt": [0.002515616907482375, 9.777082518576458e-05, 0.0033453476941939985, 0.0012023169043114293, 0.04837806128381021, 0.0006500438647485969, 0.00013212273673751968, 0.050240991871809235, 0.04601834920567811, 0.012091872866217802, 0.0034034816983585076, 0.0, 0.080510310858375, 0.044113139341923076, 0.1953619634495661, 0.16580874969611772, 0.046079125664577364, 1.0569818939001577e-05, 0.027386400870953086, 0.010553964210593072, 0.00044921730490756693, 0.01652326945639421, 0.0013185849126404465, 0.010556606665327824, 0.0012498810895369362, 0.0013106575484361953, 0.011584521557145726, 0.004457821137523914, 0.0311994630531979, 0.1804955131118604, 0.002872348296673678, 8.191609677726221e-05]}  # fmt: skip

    names = sorted(name2load.keys())
    callback = AdaptiveSamplingCallback(sim_type="l2")
    # callback = AdaptiveSamplingCallback(sim_type="~l2")
    # callback = RandomSamplingCallback()
    callback.prob_map = {"code": 0.25, "math": 0.25, "orca": 0.25, "sharegpt": 0.25}
    name2probs = defaultdict(list)
    eval_steps = 100
    for _ in range(int(2000 / eval_steps)):
        for name in names:
            name2probs[name].append(callback.prob_map[name])
        new_name2prob, _ = callback._update_prob_map(name2load)
        callback.prob_map = new_name2prob
    print(f"final prob_map: {callback.prob_map}")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name in names:
        ax.plot(name2probs[name], label=name)
    ax.legend()
    ax.set_title("Sampling Probability")
    ax.set_xlabel("step")
    fig.savefig("results/dynamic-sampling_convergence.png")


if __name__ == "__main__":
    sampling_info_stats(
        "/mnt/petrelfs/zhutong/adaptive-sft-for-moe/outputs//llama_moe_dynamic_sim_better_reverse/2726375/sampling_info/data.jsonl",
        "/mnt/petrelfs/zhutong/adaptive-sft-for-moe/outputs//llama_moe_dynamic_sim_better_reverse/2726375/sampling_info/",
    )
