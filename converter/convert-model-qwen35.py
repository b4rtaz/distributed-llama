import os

base = __import__('base-model-converter')


class ArchType:
    QWEN35 = 0xABCE01
    QWEN35_STARTLUX_INSTANT = 0xABCE02


HEADER_MAGIC = 0xA00ABCE
HEADER_KEYS = {
    'version': 0,
    'arch_type': 1,
    'dim': 2,
    'hidden_dim': 3,
    'n_layers': 4,
    'n_heads': 5,
    'n_kv_heads': 6,
    'vocab_size': 7,
    'max_seq_len': 8,
    'rope_theta': 9,
    'weights_float_type': 10,
    'head_dim': 11,
    'rotary_dim': 12,
    'norm_epsilon': 13,
    'full_attention_interval': 14,
    'linear_conv_kernel_dim': 15,
    'linear_key_head_dim': 16,
    'linear_value_head_dim': 17,
    'linear_n_key_heads': 18,
    'linear_n_value_heads': 19,
}


def resolveArchitecture(folderPath, config):
    architectures = config.get('architectures', [])
    promptAttention = config.get('jev_prompt_attention')
    isQwen35Architecture = ('Qwen3_5ForConditionalGeneration' in architectures or
        'Qwen3_5ForCausalLM' in architectures or
        config.get('model_type') == 'qwen3_5_text')
    isQwen35 = isQwen35Architecture and promptAttention != 'bidirectional'
    if isQwen35:
        return ArchType.QWEN35

    isStartLuxInstant = isQwen35Architecture and promptAttention == 'bidirectional'
    if isStartLuxInstant:
        decisionConfigPath = os.path.join(folderPath, 'decision_config.json')
        if not os.path.exists(decisionConfigPath):
            raise Exception('A bidirectional StartLux model requires decision_config.json')
        return ArchType.QWEN35_STARTLUX_INSTANT

    raise Exception(f'Unsupported Qwen3.5 architecture: {architectures}')


def sliceRows(start: int, end: int):
    def transform(tensor):
        return tensor[start:end].contiguous()
    return transform


def splitFullAttentionQProj(part: int, nHeads: int, headDim: int, dim: int):
    def transform(tensor):
        tensor = tensor.reshape(nHeads, 2, headDim, dim)
        return tensor[:, part, :, :].reshape(nHeads * headDim, dim).contiguous()
    return transform


def splitConv(start: int, end: int):
    def transform(tensor):
        return tensor[start:end, 0, :].contiguous()
    return transform


class Qwen35ModelPlan(base.BaseModelPlan):
    headerMagic = HEADER_MAGIC
    headerKeys = HEADER_KEYS

    def __init__(self, folderPath, sourceConfig, weightsFloatType, weightMap):
        super().__init__(folderPath, sourceConfig, weightsFloatType, weightMap)
        self.archType = resolveArchitecture(folderPath, sourceConfig)
        self.textConfig = sourceConfig.get('text_config', sourceConfig)
        self.__buildHeader()
        self.__validateLayerTypes()
        self.__buildTensors()

    def __buildHeader(self):
        config = self.textConfig
        if config.get('hidden_act') != 'silu':
            raise Exception(f"Unsupported hidden activation: {config.get('hidden_act')}")
        if config.get('attention_bias', False):
            raise Exception('Attention bias is not supported')

        ropeParameters = config.get('rope_parameters', {})
        headDim = int(config.get('head_dim', config['hidden_size'] // config['num_attention_heads']))
        rotaryDim = int(headDim * float(ropeParameters.get('partial_rotary_factor', 1.0)))
        if rotaryDim <= 0 or rotaryDim > headDim or rotaryDim % 2 != 0:
            raise Exception(f'Invalid rotary dimension: {rotaryDim}')

        self.header = {
            'version': 0,
            'arch_type': self.archType,
            'dim': int(config['hidden_size']),
            'hidden_dim': int(config['intermediate_size']),
            'n_layers': int(config['num_hidden_layers']),
            'n_heads': int(config['num_attention_heads']),
            'n_kv_heads': int(config['num_key_value_heads']),
            'vocab_size': int(config['vocab_size']),
            'max_seq_len': int(config['max_position_embeddings']),
            'rope_theta': int(ropeParameters.get('rope_theta', config.get('rope_theta', 10000000))),
            'weights_float_type': self.weightsFloatType,
            'head_dim': headDim,
            'rotary_dim': rotaryDim,
            'norm_epsilon': base.parseRmsNormEpsilon(float(config['rms_norm_eps'])),
            'full_attention_interval': int(config.get('full_attention_interval', 4)),
            'linear_conv_kernel_dim': int(config.get('linear_conv_kernel_dim', 4)),
            'linear_key_head_dim': int(config.get('linear_key_head_dim', 128)),
            'linear_value_head_dim': int(config.get('linear_value_head_dim', 128)),
            'linear_n_key_heads': int(config.get('linear_num_key_heads', 16)),
            'linear_n_value_heads': int(config.get('linear_num_value_heads', 32)),
        }

    def __validateLayerTypes(self):
        nLayers = self.header['n_layers']
        interval = self.header['full_attention_interval']
        self.layerTypes = self.textConfig.get('layer_types')
        if self.layerTypes is None:
            self.layerTypes = [
                'full_attention' if (layerIndex + 1) % interval == 0 else 'linear_attention'
                for layerIndex in range(nLayers)
            ]
        if len(self.layerTypes) != nLayers:
            raise Exception('layer_types length does not match num_hidden_layers')
        for layerIndex, layerType in enumerate(self.layerTypes):
            expected = 'full_attention' if (layerIndex + 1) % interval == 0 else 'linear_attention'
            if layerType != expected:
                raise Exception(
                    f'Layer {layerIndex} is {layerType}, but the current graph expects {expected}; '
                    'the header currently supports only full_attention_interval'
                )

    def __buildTensors(self):
        h = self.header
        wt = self.weightsFloatType
        conditionalPrefix = 'model.language_model'
        prefix = conditionalPrefix if f'{conditionalPrefix}.embed_tokens.weight' in self.weightMap else 'model'
        dim = h['dim']
        hiddenDim = h['hidden_dim']
        headDim = h['head_dim']
        qDim = h['n_heads'] * headDim
        kvDim = h['n_kv_heads'] * headDim
        linearKeyDim = h['linear_n_key_heads'] * h['linear_key_head_dim']
        linearValueDim = h['linear_n_value_heads'] * h['linear_value_head_dim']
        linearQkvDim = linearKeyDim * 2 + linearValueDim
        convKernel = h['linear_conv_kernel_dim']

        self.addTensor('embedding', base.FloatType.F32, f'{prefix}.embed_tokens.weight',
            (h['vocab_size'], dim))
        for layerIndex, layerType in enumerate(self.layerTypes):
            layer = f'{prefix}.layers.{layerIndex}'
            if layerType == 'full_attention':
                qProj = f'{layer}.self_attn.q_proj.weight'
                qProjShape = (qDim * 2, dim)
                self.addTensor('block_matmul_q', wt, qProj, qProjShape, (qDim, dim),
                    splitFullAttentionQProj(0, h['n_heads'], headDim, dim))
                self.addTensor('block_matmul_gate', wt, qProj, qProjShape, (qDim, dim),
                    splitFullAttentionQProj(1, h['n_heads'], headDim, dim))
                self.addTensor('block_matmul_k', wt, f'{layer}.self_attn.k_proj.weight', (kvDim, dim))
                self.addTensor('block_matmul_v', wt, f'{layer}.self_attn.v_proj.weight', (kvDim, dim))
                self.addTensor('block_matmul_wo', wt, f'{layer}.self_attn.o_proj.weight', (dim, qDim))
                self.addTensor('block_norm_q', base.FloatType.F32,
                    f'{layer}.self_attn.q_norm.weight', (headDim,))
                self.addTensor('block_norm_k', base.FloatType.F32,
                    f'{layer}.self_attn.k_norm.weight', (headDim,))
            else:
                qkv = f'{layer}.linear_attn.in_proj_qkv.weight'
                qkvShape = (linearQkvDim, dim)
                self.addTensor('block_linear_matmul_q', wt, qkv, qkvShape,
                    (linearKeyDim, dim), sliceRows(0, linearKeyDim))
                self.addTensor('block_linear_matmul_k', wt, qkv, qkvShape,
                    (linearKeyDim, dim), sliceRows(linearKeyDim, linearKeyDim * 2))
                self.addTensor('block_linear_matmul_v', wt, qkv, qkvShape,
                    (linearValueDim, dim), sliceRows(linearKeyDim * 2, linearQkvDim))
                self.addTensor('block_linear_matmul_z', wt,
                    f'{layer}.linear_attn.in_proj_z.weight', (linearValueDim, dim))
                self.addTensor('block_linear_matmul_a', wt,
                    f'{layer}.linear_attn.in_proj_a.weight', (h['linear_n_value_heads'], dim))
                self.addTensor('block_linear_matmul_b', wt,
                    f'{layer}.linear_attn.in_proj_b.weight', (h['linear_n_value_heads'], dim))

                conv = f'{layer}.linear_attn.conv1d.weight'
                convShape = (linearQkvDim, 1, convKernel)
                self.addTensor('block_linear_conv_q', base.FloatType.F32, conv, convShape,
                    (linearKeyDim, convKernel), splitConv(0, linearKeyDim))
                self.addTensor('block_linear_conv_k', base.FloatType.F32, conv, convShape,
                    (linearKeyDim, convKernel), splitConv(linearKeyDim, linearKeyDim * 2))
                self.addTensor('block_linear_conv_v', base.FloatType.F32, conv, convShape,
                    (linearValueDim, convKernel), splitConv(linearKeyDim * 2, linearQkvDim))
                self.addTensor('block_linear_delta.A_log', base.FloatType.F32,
                    f'{layer}.linear_attn.A_log', (h['linear_n_value_heads'],))
                self.addTensor('block_linear_delta.dt_bias', base.FloatType.F32,
                    f'{layer}.linear_attn.dt_bias', (h['linear_n_value_heads'],))
                self.addTensor('block_linear_norm', base.FloatType.F32,
                    f'{layer}.linear_attn.norm.weight', (h['linear_value_head_dim'],))
                self.addTensor('block_matmul_wo', wt, f'{layer}.linear_attn.out_proj.weight',
                    (dim, linearValueDim))

            self.addTensor('block_matmul_w1', wt, f'{layer}.mlp.gate_proj.weight', (hiddenDim, dim))
            self.addTensor('block_matmul_w2', wt, f'{layer}.mlp.down_proj.weight', (dim, hiddenDim))
            self.addTensor('block_matmul_w3', wt, f'{layer}.mlp.up_proj.weight', (hiddenDim, dim))
            self.addTensor('block_norm_0', base.FloatType.F32,
                f'{layer}.input_layernorm.weight', (dim,))
            self.addTensor('block_norm_1', base.FloatType.F32,
                f'{layer}.post_attention_layernorm.weight', (dim,))

        self.addTensor('final_norm', base.FloatType.F32, f'{prefix}.norm.weight', (dim,))
        self.addTensor('final_matmul_logits', wt,
            ['lm_head.weight', f'{prefix}.embed_tokens.weight'], (h['vocab_size'], dim))


if __name__ == '__main__':
    base.runConverter(Qwen35ModelPlan, 'Qwen3.5 Hugging Face', 'qwen35_9b')
