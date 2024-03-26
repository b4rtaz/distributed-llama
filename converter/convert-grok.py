import torch
from writer import writeTensor, writeHeader 

def _writeTensor(outFile, tensor, targetFloatType):
    print(f'Writing tensor {tensor.shape}...')
    writeTensor(outFile, tensor, targetFloatType)

#### pytorch_model1.bin
model = torch.load('/Users/b4rtaz/Dev/llama.cpp-grok/_weights/pytorch_model1.bin', map_location='cpu')

targetFloatType = 'q40'
outputFileName = f'dllama_grok_{targetFloatType}.bin'
outFile = open(outputFileName, 'wb')

params = {
    'dim': 6144,
    'hidden_dim': 32768,
    'n_layers': 1, # 64
    'n_heads': 48,
    'n_kv_heads': 8,
    'n_experts': 8,
    'n_active_experts': 2,
    'vocab_size': 131072,
    'max_seq_len': 8192,
}
writeHeader(outFile, params)

_writeTensor(outFile, model['transformer.in_out_embed.weight'], 'f32')

_writeTensor(outFile, model['transformer.decoder_layer.0.rms_norm.weight'], 'f32')
_writeTensor(outFile, model['transformer.decoder_layer.0.rms_norm_1.weight'], 'f32')
_writeTensor(outFile, model['transformer.decoder_layer.0.rms_norm_2.weight'], 'f32')
_writeTensor(outFile, model['transformer.decoder_layer.0.rms_norm_3.weight'], 'f32')

_writeTensor(outFile, model['transformer.decoder_layer.0.multi_head_attention.query.weight'], 'q40')
_writeTensor(outFile, model['transformer.decoder_layer.0.multi_head_attention.key.weight'], 'q40')
_writeTensor(outFile, model['transformer.decoder_layer.0.multi_head_attention.value.weight'], 'q40')
_writeTensor(outFile, model['transformer.decoder_layer.0.multi_head_attention.linear.weight'], 'q40')

_writeTensor(outFile, model['transformer.decoder_layer.0.router.weight'], 'q40')
for index in range(params['n_experts']):
    _writeTensor(outFile, model[f'transformer.decoder_layer.0.moe.{index}.linear_v.weight'], 'q40') # up
    _writeTensor(outFile, model[f'transformer.decoder_layer.0.moe.{index}.linear.weight'], 'q40') # gate
    _writeTensor(outFile, model[f'transformer.decoder_layer.0.moe.{index}.linear_1.weight'], 'q40') # down

del model

#### pytorch_model-00019-of-00019.bin
model = torch.load('/Users/b4rtaz/Dev/llama.cpp-grok/_weights/pytorch_model-00019-of-00019.bin', map_location='cpu')

_writeTensor(outFile, model['transformer.rms_norm.weight'], 'f32') # rmsFinalNorm

del model

#### pytorch_model1.bin
model = torch.load('/Users/b4rtaz/Dev/llama.cpp-grok/_weights/pytorch_model1.bin', map_location='cpu')

_writeTensor(outFile, model['lm_head.weight'], 'q40')

del model

outFile.close()
print(f'Saved {outputFileName}')
