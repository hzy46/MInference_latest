# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from ..ops.streaming_kernel import tri_shape_kernel

try:
    from flash_attn import flash_attn_func
except ImportError:
    from ..ops.flash_attn_triton import _flash_attn_triton_decoding as flash_attn_func

import copy

import numpy as np
import torch

from ..modules.flexprefill import flexprefill_forward
from ..modules.minference_forward import minference_prefill_forward
from ..modules.xattention import xattention_forward

g = {"timer": []}

if "TRACK_ATTENTION" in os.environ:
    track_attention = True
    if "TRACK_ATTENTION_LAYER_NUM" in os.environ:
        track_attention_layer_num = int(os.environ["TRACK_ATTENTION_LAYER_NUM"])
    else:
        print("set track_attention_layer_num to 32")
        track_attention_layer_num = 32
else:
    track_attention = False
    track_attention_layer_num = None

print("track_attention", track_attention)


def tri_mix_forward(query_states, key_states, value_states, prefill_kwargs):
    starting_layer = prefill_kwargs["attn_forward_config"].get("starting_layer", 0)
    layer_idx = prefill_kwargs["layer_idx"]

    global g, track_attention, track_attention_layer_num

    if track_attention:
        layer_idx = config["layer_idx"]
        if layer_idx == 0:
            g["timer"] = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for i in range(track_attention_layer_num)
            ]
        start_event, end_event = g["timer"][layer_idx]
        start_event.record()

    bsz, head_num, q_len, head_dim = query_states.shape
    if layer_idx < starting_layer:
        # flash attention
        result = flash_attn_func(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            0.0,
            softmax_scale=None,
            causal=q_len != 1,
        ).transpose(1, 2)
    else:
        result = tri_shape_kernel(
            query_states, key_states, value_states, prefill_kwargs
        )

    if track_attention:
        torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()
        if layer_idx == track_attention_layer_num - 1:
            time_ms_list = []
            for layer_idx in range(track_attention_layer_num):
                start_event, end_event = g["timer"][layer_idx]
                elapsed_time_ms = start_event.elapsed_time(end_event)
                time_ms_list.append(elapsed_time_ms)
            import numpy as np

            print("Average Attn {:.1f} ms".format(np.mean(time_ms_list)))

    return result


def minference_mix_forward(q, k, v, prefill_kwargs):
    layer_idx = prefill_kwargs["layer_idx"]
    starting_layer = prefill_kwargs["attn_forward_config"].get("starting_layer", 0)

    if track_attention:
        layer_idx = config["layer_idx"]
        if layer_idx == 0:
            g["timer"] = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for i in range(track_attention_layer_num)
            ]
        start_event, end_event = g["timer"][layer_idx]
        start_event.record()

    if layer_idx < starting_layer:
        # print("layer", layer_idx, "minference foward")
        minference_prefill_kwargs = copy.deepcopy(prefill_kwargs)
        minference_prefill_kwargs["attn_forward_config"]["starting_layer"] = 0
        result = minference_prefill_forward(q, k, v, minference_prefill_kwargs)
    else:
        # print("layer", layer_idx, "tri forward")
        result = tri_shape_kernel(q, k, v, prefill_kwargs)

    if track_attention:
        torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()
        if layer_idx == track_attention_layer_num - 1:
            time_ms_list = []
            for layer_idx in range(track_attention_layer_num):
                start_event, end_event = g["timer"][layer_idx]
                elapsed_time_ms = start_event.elapsed_time(end_event)
                time_ms_list.append(elapsed_time_ms)
            import numpy as np

            print("Average Attn {:.1f} ms".format(np.mean(time_ms_list)))
    return result


def minference_mix_per_layer_forward(q, k, v, prefill_kwargs):
    tri_layer_idx_list = prefill_kwargs["attn_forward_config"].get(
        "tri_layer_idx_list", []
    )
    layer_idx = prefill_kwargs["layer_idx"]
    if layer_idx == 0:
        print("tri_layer_idx_list:", tri_layer_idx_list)

    if not (layer_idx in tri_layer_idx_list):
        minference_prefill_kwargs = copy.deepcopy(prefill_kwargs)
        minference_prefill_kwargs["attn_forward_config"]["starting_layer"] = 0
        result = minference_prefill_forward(q, k, v, minference_prefill_kwargs)
    else:
        result = tri_shape_kernel(q, k, v, prefill_kwargs)
    return result


def flexprefill_mix_forward(q, k, v, prefill_kwargs):
    layer_idx = prefill_kwargs["layer_idx"]
    starting_layer = prefill_kwargs["attn_forward_config"].get("starting_layer", 0)

    if track_attention:
        layer_idx = config["layer_idx"]
        if layer_idx == 0:
            g["timer"] = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for i in range(track_attention_layer_num)
            ]
        start_event, end_event = g["timer"][layer_idx]
        start_event.record()

    if layer_idx < starting_layer:
        result = flexprefill_forward(q, k, v, prefill_kwargs)
    else:
        result = tri_shape_kernel(q, k, v, prefill_kwargs)

    if track_attention:
        torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()
        if layer_idx == track_attention_layer_num - 1:
            time_ms_list = []
            for layer_idx in range(track_attention_layer_num):
                start_event, end_event = g["timer"][layer_idx]
                elapsed_time_ms = start_event.elapsed_time(end_event)
                time_ms_list.append(elapsed_time_ms)
            import numpy as np

            print("Average Attn {:.1f} ms".format(np.mean(time_ms_list)))
    return result


def flexprefill_mix_per_layer_forward(q, k, v, prefill_kwargs):
    tri_layer_idx_list = prefill_kwargs["attn_forward_config"].get(
        "tri_layer_idx_list", []
    )
    layer_idx = prefill_kwargs["layer_idx"]
    if layer_idx == 0:
        print("tri_layer_idx_list:", tri_layer_idx_list)

    if not (layer_idx in tri_layer_idx_list):
        result = flexprefill_forward(q, k, v, prefill_kwargs)
    else:
        result = tri_shape_kernel(q, k, v, prefill_kwargs)
    return result


def xattention_mix_forward(q, k, v, prefill_kwargs):
    layer_idx = prefill_kwargs["layer_idx"]
    starting_layer = prefill_kwargs["attn_forward_config"].get("starting_layer", 0)

    if layer_idx < starting_layer:
        # print("layer", layer_idx, "xattention foward")
        xattention_prefill_kwargs = copy.deepcopy(prefill_kwargs)
        xattention_prefill_kwargs["attn_forward_config"]["starting_layer"] = 0
        result = xattention_forward(q, k, v, xattention_prefill_kwargs)
    else:
        # print("layer", layer_idx, "tri forward")
        result = tri_shape_kernel(q, k, v, prefill_kwargs)
    return result


def xattention_mix_per_layer_forward(q, k, v, prefill_kwargs):
    tri_layer_idx_list = prefill_kwargs["attn_forward_config"].get(
        "tri_layer_idx_list", []
    )
    layer_idx = prefill_kwargs["layer_idx"]
    if layer_idx == 0:
        print("tri_layer_idx_list:", tri_layer_idx_list)

    if not (layer_idx in tri_layer_idx_list):
        xattention_prefill_kwargs = copy.deepcopy(prefill_kwargs)
        xattention_prefill_kwargs["attn_forward_config"]["starting_layer"] = 0
        result = xattention_forward(q, k, v, xattention_prefill_kwargs)
    else:
        result = tri_shape_kernel(q, k, v, prefill_kwargs)
    return result


def tri_mix_per_layer_forward(query_states, key_states, value_states, prefill_kwargs):
    tri_layer_idx_list = prefill_kwargs["attn_forward_config"].get(
        "tri_layer_idx_list", []
    )
    layer_idx = prefill_kwargs["layer_idx"]
    if layer_idx == 0:
        print("tri_layer_idx_list:", tri_layer_idx_list)

    bsz, head_num, q_len, head_dim = query_states.shape
    if layer_idx in tri_layer_idx_list:
        result = tri_shape_kernel(
            query_states, key_states, value_states, prefill_kwargs
        )
    else:
        # flash attention
        result = flash_attn_func(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            0.0,
            softmax_scale=None,
            causal=q_len != 1,
        ).transpose(1, 2)

    return result
