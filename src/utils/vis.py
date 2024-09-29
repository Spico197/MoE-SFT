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
        "auto",
        model_dir,
        trust_remote_code=True,
        padding_side="right",
        torch_dtype=torch.bfloat16,
        attn_impl="flash_attention_2",
        model_max_length=2048,
        use_fast=True,
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
                print(gate_load)
                # (num_layers, num_experts)
                gate_load = torch.stack(gate_load, dim=0).detach().cpu().numpy()
                gate_load_list.append(gate_load)
                if hasattr(outs, "expert2tokens"):
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
        cb = AdaptiveSamplingCallback()
        new_weights = cb._update_weights(ori_weights, _delta_vec)
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

    # fmt: off
    # dynamic: 10.0, 5e-2
    # name2load = {"code": [0.14031716417910448, 0.1310634328358209, 0.12651119402985075, 0.10993470149253731, 0.12196828358208955, 0.12552238805970148, 0.12791977611940297, 0.11676305970149255], "orca": [0.15106234655836084, 0.11803640166095838, 0.12349968175067437, 0.12884551268450883, 0.11344072985178673, 0.1383778377231534, 0.11733170672566907, 0.1094057830448883], "math": [0.16001617686708006, 0.10756444371505268, 0.11391210568886491, 0.114803005615014, 0.11676650216277679, 0.1177863481308685, 0.13630182751708533, 0.13284959030325763], "sharegpt": [0.15440024978412215, 0.113654214863131, 0.12914741653941664, 0.12104040941178769, 0.11470799162832905, 0.13593110446537907, 0.12316259873058931, 0.10795601457724527]}  # fmt: skip
    # moduleformer: 10.0, 8e-1
    # name2load = {"code": [0.0009968102073365233, 0.0001812382195157315, 0.0009786863853849502, 0.0008518196317239381, 0.13205016673916198, 0.000724952878062926, 5.437146585471945e-05, 0.01346599971001885, 0.08014354066985646, 5.437146585471945e-05, 0.01904813687110338, 0.0, 0.0007793243439176454, 0.000724952878062926, 0.09194214876033059, 0.23983253588516748, 0.19155067420617664, 0.0, 0.07307525010874294, 0.023071625344352618, 0.0006343337683050603, 0.0178882122662027, 0.005274032187907787, 0.03858561693489924, 1.8123821951573152e-05, 0.00028998115122517043, 0.019120632158909672, 0.0003987240829346093, 0.0016855154414963031, 0.04594388864723793, 0.0005437146585471945, 9.061910975786575e-05], "orca": [0.0020665409020751155, 8.574858514834504e-06, 0.004073057794546389, 0.002057966043560281, 0.027208026067569883, 0.01708111816155033, 0.0006945635397015948, 0.0287772251757846, 0.043568856113874115, 0.028219859372320355, 0.004115932087120562, 0.0, 0.11049562682215741, 0.0468959012176299, 0.20437317784256556, 0.1614903104098782, 0.01911335962956611, 0.0, 0.015100325844623562, 0.009715314697307492, 0.0004544675012862287, 0.022629051620648252, 0.0011490310409878236, 0.0032927456696964495, 0.001020408163265306, 0.0021179900531641226, 0.010049734179386038, 0.009783913565426168, 0.01661807580174927, 0.19773623735208365, 0.0039358600583090375, 0.006156748413651173], "math": [0.03253549581906953, 7.014984005836466e-05, 0.002118525169762613, 0.0012065772490038721, 0.11881979909085806, 0.003928391043268421, 0.03210056681070767, 0.045218586901621866, 0.06749817610415848, 0.0010803075368988157, 0.01540490487681688, 0.0, 0.018126718671081427, 0.006930804197766429, 0.16879454514843703, 0.2359980919243504, 0.027821426567147426, 0.0, 0.0370110556147932, 0.017397160334474436, 0.0002525394242101128, 0.015909983725237106, 0.0015713564173073684, 0.02965935237667658, 0.00023850945619843984, 0.0005191088164318985, 0.005906616532914305, 0.0018379258095291542, 0.0071693136539648684, 0.10159099837252371, 0.002665693922217857, 0.0006173185925136091], "sharegpt": [0.002515616907482375, 9.777082518576458e-05, 0.0033453476941939985, 0.0012023169043114293, 0.04837806128381021, 0.0006500438647485969, 0.00013212273673751968, 0.050240991871809235, 0.04601834920567811, 0.012091872866217802, 0.0034034816983585076, 0.0, 0.080510310858375, 0.044113139341923076, 0.1953619634495661, 0.16580874969611772, 0.046079125664577364, 1.0569818939001577e-05, 0.027386400870953086, 0.010553964210593072, 0.00044921730490756693, 0.01652326945639421, 0.0013185849126404465, 0.010556606665327824, 0.0012498810895369362, 0.0013106575484361953, 0.011584521557145726, 0.004457821137523914, 0.0311994630531979, 0.1804955131118604, 0.002872348296673678, 8.191609677726221e-05]}  # fmt: skip
    # olmoe uniform final
    # name2load = {'sharegpt': [0.008855465980410182, 0.016250229773377808, 0.019217588823822913, 0.009707596964365435, 0.011326514534807389, 0.030788319634463393, 0.013091174076311026, 0.007340274677660778, 0.04539862398571468, 0.011723694230718736, 0.016196397153436102, 0.013408261337674962, 0.014929361098710648, 0.014199994748037084, 0.006815734880911744, 0.009077361413828422, 0.007380320894934485, 0.034847430477140846, 0.013909823796644002, 0.01839500013129908, 0.021215304219952213, 0.01594102045639558, 0.014981880727922066, 0.012718941204275103, 0.024791890969249768, 0.014773115201806682, 0.008713662981539354, 0.014605708883695287, 0.008029594811060636, 0.014245292928231932, 0.04248444105984613, 0.011142695832567426, 0.021563903258842998, 0.01688046532391482, 0.022005724639584053, 0.012960531498647624, 0.01904361755206009, 0.028983613875686046, 0.006146766103831309, 0.003965888500827186, 0.009555290039652324, 0.00874648774979649, 0.020024421627583316, 0.009580893358892891, 0.02699180693784303, 0.009223759880255249, 0.017172605761403333, 0.0134949187258738, 0.011447966177358791, 0.00796985373283265, 0.012464221002599725, 0.019423728368477725, 0.014951025445760359, 0.013297313620965841, 0.013080013655103601, 0.01577755311047505, 0.016510201937974325, 0.011386255613035377, 0.02562170111079017, 0.006184186339644445, 0.018108768152096853, 0.01905346498253723, 0.009318951708200943, 0.01256138231664085], 'code': [0.008684385796944598, 0.037438080173167396, 0.005760104899471341, 0.017014944011988515, 0.0029190775506805987, 0.015282229530033721, 0.01780064937767973, 0.002752570453315573, 0.08977334221371189, 0.015063688964742127, 0.002029305249136245, 0.014397660575282025, 0.009901968946426346, 0.0018627981517712197, 7.284685509719853e-05, 0.0022426424676351836, 0.00959497148565958, 0.08941431128501856, 0.00700890812970903, 0.018211713774299636, 0.012342338592182495, 0.007123381759147486, 0.009652208300378806, 0.0053438371560587785, 0.024065478916038805, 0.0003069974607667653, 0.000671231736252758, 0.0023935395246222377, 0.0013372601257128588, 0.032739458019398086, 0.08691150147775051, 0.012472422261998921, 0.017160637722182914, 0.01608354493610291, 0.03151146817633103, 0.023800108229613294, 0.04918723723098698, 0.02316009657411648, 0.0007492819381426136, 0.0003746409690713068, 0.005817341714190569, 0.004136660700162346, 0.0022946759355617543, 0.01236835532614578, 0.07116097073637766, 0.0005983848811555594, 0.0023675227906589527, 0.0027785871872788586, 0.0017171044415768227, 7.805020188985558e-05, 0.004313574491112685, 0.002856637389168714, 0.008684385796944598, 0.027312367314656794, 0.0029190775506805987, 0.005770511593056656, 0.004334387878283313, 0.018268950589018864, 0.05789243641510221, 0.0009105856887149817, 0.032765474753361376, 0.0019044249261124763, 0.0012800233109936316, 0.002856637389168714], 'math': [0.006854677979613917, 0.023770713512897897, 0.011944080633221343, 0.04263353453675759, 0.005520044416604976, 0.019976795171117816, 0.013011787483628495, 0.0022920107055406868, 0.03701383748078128, 0.012093559592278343, 0.006367091851261318, 0.013150589374181424, 0.011876459199362222, 0.005359888389043903, 0.0015944422299413472, 0.004733500370138375, 0.004729941347303684, 0.09670932748704515, 0.0060396617504697905, 0.017343118273446843, 0.013410398041113832, 0.00834590854734924, 0.0091858379363362, 0.005993394453618814, 0.02689909458459085, 0.0018151016456921585, 0.010353197426114685, 0.0032600649165765044, 0.0060040715221228854, 0.020624537327031487, 0.04169395250839929, 0.008004242355218951, 0.015819856500199306, 0.021033824953020898, 0.06154262285746825, 0.025792238483002106, 0.03586783212801093, 0.029710722623996353, 0.006509452764648938, 0.0043775980866693235, 0.004544872159899778, 0.004128466488240988, 0.0038864529354820337, 0.0091858379363362, 0.04506078811001651, 0.007253288537099254, 0.007082455441034109, 0.008716046922157052, 0.017442770912818175, 0.0013239564945048687, 0.007356500199305277, 0.0091858379363362, 0.010851460622971355, 0.03826305449575764, 0.008061186720573998, 0.011111269289903762, 0.00953818119697056, 0.017581572803371106, 0.0365938727862878, 0.002722652468538238, 0.020660127555378393, 0.008370821707192073, 0.004729941347303684, 0.00708957348670349], 'orca': [0.0268433723181027, 0.015692860744668736, 0.016971794329586672, 0.010339232558517819, 0.011469736649478667, 0.018378824601070317, 0.021965531824910126, 0.012484343738308635, 0.03318720822421393, 0.008145322640987687, 0.01628047887827968, 0.012665305724091936, 0.019281601249247684, 0.01489174813344828, 0.010924817411389624, 0.011296907786651917, 0.007844397091595231, 0.02389918180783057, 0.00958691868503668, 0.013279356507311676, 0.018212095580460986, 0.01589008897636514, 0.012185451469655316, 0.010526294386518534, 0.027453356539844168, 0.012533142476047952, 0.006223872342502074, 0.010131837923125721, 0.009489321209558045, 0.009867511427037753, 0.024006945687004896, 0.021526343185256272, 0.012230183645916357, 0.011357906208826063, 0.03584673943100671, 0.019574393675683588, 0.032878149551864924, 0.02077606259251427, 0.0050161035834539736, 0.0043369878165818105, 0.011343673243652095, 0.015723359955755807, 0.01573555964019064, 0.025169982269791953, 0.01646144086406298, 0.009351058119296646, 0.017069391805065306, 0.013592481741138961, 0.012087853994176683, 0.014539990565577369, 0.010678790441953901, 0.008232753712770628, 0.01393407290531418, 0.015349236299754378, 0.013234624331050635, 0.015465133301885257, 0.015820957431234444, 0.031418253981163685, 0.019535761341639958, 0.009511687297688566, 0.01977568846885827, 0.01612391626136604, 0.011211509995608111, 0.013141093417050276]}
    # olmoe init
    name2load = {"code": [1.09859018e-02, 3.93439177e-02, 6.79040290e-03, 2.01448619e-02, 3.35478238e-03, 9.32063636e-03, 1.44700252e-02, 5.18172412e-03, 9.22524736e-02, 1.08080579e-02, 2.19071332e-03, 1.01613529e-02, 1.11314105e-02, 2.26346763e-03, 8.89219427e-05, 2.61915540e-03, 9.14279247e-03, 9.41683373e-02, 5.92543491e-03, 1.45185281e-02, 1.16972774e-02, 7.17034211e-03, 8.94878096e-03, 4.72903059e-03, 1.87786975e-02, 1.61676259e-03, 1.26107482e-03, 2.39280864e-03, 1.78652267e-03, 3.67409300e-02, 9.44027679e-02, 1.21095518e-02, 1.53673285e-02, 1.60948716e-02, 2.09774947e-02, 1.85604346e-02, 5.42423850e-02, 1.37829011e-02, 1.72993598e-03, 9.45806118e-04, 5.69100433e-03, 4.83412016e-03, 3.40328526e-03, 1.33140400e-02, 6.20998513e-02, 1.85927698e-04, 2.44939533e-03, 2.98292699e-03, 1.01047662e-03, 7.43710794e-04, 2.82125073e-03, 4.76944965e-03, 1.86089375e-02, 2.08885727e-02, 2.58682015e-03, 5.52124426e-03, 3.81555972e-03, 1.68790015e-02, 7.15417448e-02, 3.55687771e-04, 4.34747462e-02, 1.04281187e-03, 1.34191295e-03, 1.43891871e-03], "math": [0.008351  , 0.02324349, 0.01596002, 0.06318141, 0.0067232 , 0.00518625, 0.01020594, 0.01412023, 0.0266278 , 0.00685948, 0.0092444 , 0.00698819, 0.00996366, 0.00961538, 0.00224107, 0.00941096, 0.00354331, 0.10296033, 0.00125681, 0.00645064, 0.01319655, 0.01639915, 0.00705633, 0.00565566, 0.01803452, 0.00610993, 0.01627801, 0.00398243, 0.01170503, 0.02229709, 0.04911417, 0.0079043 , 0.0092974 , 0.0138401 , 0.05012871, 0.01894306, 0.03701545, 0.01078892, 0.00952453, 0.00875227, 0.00462598, 0.00462598, 0.00236978, 0.00884313, 0.02592368, 0.00486069, 0.00794216, 0.01004694, 0.01442308, 0.00193822, 0.00311175, 0.01698213, 0.02241066, 0.03687159, 0.00941853, 0.01320412, 0.01327226, 0.0138931 , 0.05476227, 0.00227892, 0.03142035, 0.00630678, 0.00595094, 0.00635978], "orca": [0.02702177, 0.01659798, 0.01970451, 0.01652411, 0.01420295, 0.00551711, 0.01760886, 0.02944401, 0.03629471, 0.00407854, 0.01977838, 0.00966952, 0.01391913, 0.0201283 , 0.00999611, 0.0175661 , 0.00731726, 0.04082426, 0.00171851, 0.00479782, 0.02062208, 0.02653966, 0.00958009, 0.01013997, 0.01824261, 0.01928072, 0.00926905, 0.01093313, 0.01497278, 0.01037325, 0.03882193, 0.01935459, 0.00341757, 0.00522551, 0.03839813, 0.01773328, 0.03348367, 0.00666407, 0.00625972, 0.00544712, 0.00917574, 0.01208787, 0.0074339 , 0.02338647, 0.0083126 , 0.00625194, 0.01857309, 0.01330871, 0.00976283, 0.01339425, 0.00696345, 0.00556376, 0.01681182, 0.01526439, 0.01567263, 0.01818429, 0.01998056, 0.02991446, 0.02455288, 0.00873639, 0.02104977, 0.01290824, 0.01289269, 0.01234837], "sharegpt": [0.01160889, 0.01873501, 0.02055463, 0.01620191, 0.01113448, 0.0136267 , 0.01225425, 0.0267666 , 0.04718126, 0.00802416, 0.01615484, 0.0087587 , 0.01186654, 0.01631092, 0.01067122, 0.01401689, 0.00720663, 0.04937249, 0.00578587, 0.00874259, 0.01998236, 0.02662044, 0.00962329, 0.0115705 , 0.01545499, 0.02391393, 0.0123137 , 0.01237688, 0.01063777, 0.01725108, 0.04964623, 0.01211551, 0.01010762, 0.00831153, 0.01980151, 0.0156123 , 0.02539043, 0.01216382, 0.00842921, 0.00456825, 0.00907208, 0.00851344, 0.01509206, 0.01169065, 0.01953644, 0.00696757, 0.01505985, 0.01340126, 0.00649439, 0.00965426, 0.00650802, 0.0195674 , 0.01992414, 0.01426214, 0.01404166, 0.01744059, 0.01890471, 0.01164977, 0.03671319, 0.00346458, 0.02420006, 0.01557143, 0.01030704, 0.01109732]}
    # fmt: on

    names = sorted(name2load.keys())
    # callback = AdaptiveSamplingCallback(sim_type="~l2")
    callback = AdaptiveSamplingCallback(sim_type="l2", c=0.3)
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
    fig.savefig("results/olmoe-sampling_convergence.png")


def simplify_load():
    # fmt: off
    data = {'eval_sharegpt_loss': 0.33769574761390686, 'eval_sharegpt_gate_load': [337.225, 618.825, 731.825, 369.675, 431.325, 1172.45, 498.525, 279.525, 1728.825, 446.45, 616.775, 510.6, 568.525, 540.75, 259.55, 345.675, 281.05, 1327.025, 529.7, 700.5, 807.9, 607.05, 570.525, 484.35, 944.1, 562.575, 331.825, 556.2, 305.775, 542.475, 1617.85, 424.325, 821.175, 642.825, 838.0, 493.55, 725.2, 1103.725, 234.075, 151.025, 363.875, 333.075, 762.55, 364.85, 1027.875, 351.25, 653.95, 513.9, 435.95, 303.5, 474.65, 739.675, 569.35, 506.375, 498.1, 600.825, 628.725, 433.6, 975.7, 235.5, 689.6, 725.575, 354.875, 478.35], 'eval_sharegpt_runtime': 26.6725, 'eval_sharegpt_samples_per_second': 37.492, 'eval_sharegpt_steps_per_second': 2.362, 'epoch': 1.0, 'eval_code_loss': 0.259382426738739, 'eval_code_gate_load': [41.725, 179.875, 27.675, 81.75, 14.025, 73.425, 85.525, 13.225, 431.325, 72.375, 9.75, 69.175, 47.575, 8.95, 0.35, 10.775, 46.1, 429.6, 33.675, 87.5, 59.3, 34.225, 46.375, 25.675, 115.625, 1.475, 3.225, 11.5, 6.425, 157.3, 417.575, 59.925, 82.45, 77.275, 151.4, 114.35, 236.325, 111.275, 3.6, 1.8, 27.95, 19.875, 11.025, 59.425, 341.9, 2.875, 11.375, 13.35, 8.25, 0.375, 20.725, 13.725, 41.725, 131.225, 14.025, 27.725, 20.825, 87.775, 278.15, 4.375, 157.425, 9.15, 6.15, 13.725], 'eval_code_runtime': 4.6929, 'eval_code_samples_per_second': 213.087, 'eval_code_steps_per_second': 13.425, 'eval_math_loss': 0.2713015675544739, 'eval_math_gate_load': [48.15, 166.975, 83.9, 299.475, 38.775, 140.325, 91.4, 16.1, 260.0, 84.95, 44.725, 92.375, 83.425, 37.65, 11.2, 33.25, 33.225, 679.325, 42.425, 121.825, 94.2, 58.625, 64.525, 42.1, 188.95, 12.75, 72.725, 22.9, 42.175, 144.875, 292.875, 56.225, 111.125, 147.75, 432.3, 181.175, 251.95, 208.7, 45.725, 30.75, 31.925, 29.0, 27.3, 64.525, 316.525, 50.95, 49.75, 61.225, 122.525, 9.3, 51.675, 64.525, 76.225, 268.775, 56.625, 78.05, 67.0, 123.5, 257.05, 19.125, 145.125, 58.8, 33.225, 49.8], 'eval_math_runtime': 4.6256, 'eval_math_samples_per_second': 216.188, 'eval_math_steps_per_second': 13.62, 'eval_orca_loss': 0.2595311105251312, 'eval_orca_gate_load': [330.05, 192.95, 208.675, 127.125, 141.025, 225.975, 270.075, 153.5, 408.05, 100.15, 200.175, 155.725, 237.075, 183.1, 134.325, 138.9, 96.45, 293.85, 117.875, 163.275, 223.925, 195.375, 149.825, 129.425, 337.55, 154.1, 76.525, 124.575, 116.675, 121.325, 295.175, 264.675, 150.375, 139.65, 440.75, 240.675, 404.25, 255.45, 61.675, 53.325, 139.475, 193.325, 193.475, 309.475, 202.4, 114.975, 209.875, 167.125, 148.625, 178.775, 131.3, 101.225, 171.325, 188.725, 162.725, 190.15, 194.525, 386.3, 240.2, 116.95, 243.15, 198.25, 137.85, 161.575], 'eval_orca_runtime': 4.6627, 'eval_orca_samples_per_second': 214.469, 'eval_orca_steps_per_second': 13.512, 'all_metrics': True}
    # fmt: on
    load = {}
    for k in ["sharegpt", "code", "math", "orca"]:
        sum_val = sum(data[f"eval_{k}_gate_load"])
        norm_val = [val / sum_val for val in data[f"eval_{k}_gate_load"]]
        load[k] = norm_val
    print(load)


if __name__ == "__main__":
    # sampling_info_stats(
    #     "/mnt/petrelfs/zhutong/adaptive-sft-for-moe/outputs//llama_moe_dynamic_sim_better_reverse/2726375/sampling_info/data.jsonl",
    #     "/mnt/petrelfs/zhutong/adaptive-sft-for-moe/outputs//llama_moe_dynamic_sim_better_reverse/2726375/sampling_info/",
    # )

    # gate_load_stats(
    #     "data/llama-moe-models/OLMoE-1B-7B-0924",
    #     "data/four_types_mix/dev",
    #     "results/gate_load-OLMoE-1B-7B-0924",
    #     update_strategy="l2",
    # )

    # simplify_load()

    # test_sampling_convergence()

    # sampling_info_stats(
    #     "outputs/olmoe_dynamic/3668915/sampling_info/data.jsonl",
    #     "outputs/olmoe_dynamic/3668915/sampling_info/",
    # )

    sampling_info_stats(
        "outputs/olmoe_dynamic_m200/3673738/sampling_info/data.jsonl",
        "outputs/olmoe_dynamic_m200/3673738/sampling_info/",
    )
