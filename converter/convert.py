import os
import struct
import json
import torch
import numpy as np
from pathlib import Path
from torch import nn
from model import ModelArgs, Transformer

def load_meta_model(model_path):
    params_path = os.path.join(model_path, 'params.json')
    with open(params_path) as f:
        params = json.load(f)
        print(params)

    model_paths = sorted(list(Path(model_path).glob('consolidated.*.pth')))
    models = [torch.load(p, map_location='cpu') for p in model_paths]

    def concat_weights(models):
        state_dict = {}
        for name in list(models[0]):
            tensors = [model[name] for model in models]
            if len(tensors) == 1 or len(tensors[0].shape) == 1:
                state_dict[name] = tensors[0]
                continue
            is_axis_1 = (
                name.startswith('tok_embeddings.')
                or name.endswith('.attention.wo.weight')
                or name.endswith('.feed_forward.w2.weight')
            )
            axis = 1 if is_axis_1 else 0
            state_dict[name] = torch.cat(tensors, dim=axis)
            for model in models:
                del model[name]
        return state_dict

    state_dict = concat_weights(models)
    del models

    # set ModelArgs
    config = ModelArgs()
    config.dim = params["dim"]
    config.n_layers = params["n_layers"]
    config.n_heads = params["n_heads"]
    config.n_kv_heads = params.get('n_kv_heads') or params['n_heads']
    config.multiple_of = params["multiple_of"]
    config.norm_eps = params["norm_eps"]

    config.vocab_size = state_dict['tok_embeddings.weight'].shape[0]
    config.max_seq_len = 2048

    # create a new Transformer object and set weights
    model = Transformer(config)

    model.tok_embeddings.weight = nn.Parameter(state_dict['tok_embeddings.weight'])
    model.norm.weight = nn.Parameter(state_dict['norm.weight'])

    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(state_dict[f'layers.{i}.attention_norm.weight'])
        layer.attention.wq.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wq.weight'])
        layer.attention.wk.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wk.weight'])
        layer.attention.wv.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wv.weight'])
        layer.attention.wo.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wo.weight'])
        layer.ffn_norm.weight = nn.Parameter(state_dict[f'layers.{i}.ffn_norm.weight'])
        layer.feed_forward.w1.weight = nn.Parameter(state_dict[f'layers.{i}.feed_forward.w1.weight'])
        layer.feed_forward.w2.weight = nn.Parameter(state_dict[f'layers.{i}.feed_forward.w2.weight'])
        layer.feed_forward.w3.weight = nn.Parameter(state_dict[f'layers.{i}.feed_forward.w3.weight'])

    # final classifier
    model.output.weight = nn.Parameter(state_dict['output.weight'])
    model.eval()
    return model

def save_quantized_q40(x, file):
    group_size = 32
    group_size_half = group_size // 2
    assert(x.shape[0] % group_size == 0)
    groups = x.reshape(-1, group_size)
    gmax = np.max(groups, axis=1)
    gmin = np.min(groups, axis=1)
    absMax = np.where(-gmin > gmax, gmin, gmax)

    bytes = 0
    for gi, group in enumerate(groups):
        groupAbsMax = absMax[gi]
        delta = groupAbsMax / -8
        id = (1.0 / delta) if delta else 0

        twoBytes = struct.pack('e', np.float16(delta).astype(np.float16))
        file.write(twoBytes)
        bytes += 2

        for i in range(0, group_size_half):
            x0 = group[i] * id + 8.5
            x1 = group[i + group_size_half] * id + 8.5
            xi0 = min(15, int(x0))
            xi1 = min(15, int(x1))
            b = (xi0 & 0xF) | ((xi1 & 0xF) << 4)
            byte = struct.pack('B', b)
            file.write(byte)
            bytes += 1
    print(f'Quantized {x.shape[0] * 4} bytes into {bytes} bytes')

def export_tensor(file, tensor, floatType):
    d = tensor.detach().cpu().view(-1)
    if (floatType == 'float16'):
        d = d.to(torch.float16).numpy().astype(np.float16)
        b = struct.pack(f'{len(d)}e', *d)
        file.write(b)
    elif (floatType == 'float32'):
        d = d.to(torch.float32).numpy().astype(np.float32)
        b = struct.pack(f'{len(d)}f', *d)
        file.write(b)
    elif (floatType == 'q40'):
        d = d.to(torch.float32).numpy().astype(np.float32)
        save_quantized_q40(d, file)
    else:
        raise Exception('unknown float type')

def export(model, filepath, floatType):
    out_file = open(filepath, 'wb')

    # first write out the header
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    p = model.params
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    # legacy format uses negative/positive vocab size as a shared classifier flag
    if not shared_classifier:
        p.vocab_size = -p.vocab_size
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii',
        p.dim, hidden_dim, p.n_layers, p.n_heads,
        n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)

    # next write out the embedding weights
    export_tensor(out_file, model.tok_embeddings.weight, 'float32')

    for layer in model.layers:
        export_tensor(out_file, layer.attention_norm.weight, 'float32')
        export_tensor(out_file, layer.ffn_norm.weight, 'float32')
        export_tensor(out_file, layer.attention.wq.weight, floatType)
        export_tensor(out_file, layer.attention.wk.weight, floatType)
        export_tensor(out_file, layer.attention.wv.weight, floatType)
        export_tensor(out_file, layer.attention.wo.weight, floatType)
        export_tensor(out_file, layer.feed_forward.w1.weight, floatType)
        export_tensor(out_file, layer.feed_forward.w2.weight, floatType)
        export_tensor(out_file, layer.feed_forward.w3.weight, floatType)
        print(f"Processed {layer.layer_id} layer")

    # final rmsnorm
    export_tensor(out_file, model.norm.weight, 'float32')
    # freqs_cis
    export_tensor(out_file, model.freqs_cos[:p.max_seq_len], 'float32')
    export_tensor(out_file, model.freqs_sin[:p.max_seq_len], 'float32')

    # final classifier weights
    if not shared_classifier:
        export_tensor(out_file, model.output.weight, 'float32')

    # write to binary file
    out_file.close()
    print(f"Saved {filepath}")

# targetFloatType = 'float16'
targetFloatType = 'q40'

model = load_meta_model("/Users/b4rtaz/Dev/llama.cpp/models/7B")
export(model, "llama_7b_{0}.bin".format(targetFloatType), targetFloatType)
