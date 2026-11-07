base = __import__('base-model-converter')


class ArchType:
    LLAMA = 0xABCD00
    QWEN3 = 0xABCD01
    QWEN3_MOE = 0xABCD02


HEADER_MAGIC = 0xA00ABCD
HEADER_KEYS = {
    'version': 0,
    'arch_type': 1,
    'dim': 2,
    'hidden_dim': 3,
    'n_layers': 4,
    'n_heads': 5,
    'n_kv_heads': 6,
    'n_experts': 7,
    'n_active_experts': 8,
    'vocab_size': 9,
    'max_seq_len': 10,
    'hidden_act': 11,
    'rope_theta': 12,
    'weights_float_type': 13,
    'rope_scaling_factor': 14,
    'rope_scaling_low_freq_factor': 15,
    'rope_scaling_high_freq_factory': 16,
    'rope_scaling_orig_max_seq_len': 17,
    'rope_type': 18,
    'head_dim': 19,
    'norm_epsilon': 20,
    'moe_hidden_dim': 21,
}


def parseArchType(modelType: str):
    archType = {
        'llama': ArchType.LLAMA,
        'mistral': ArchType.LLAMA,
        'qwen3': ArchType.QWEN3,
        'qwen3_moe': ArchType.QWEN3_MOE,
    }.get(modelType)
    if archType is None:
        raise Exception(f'Unsupported Llama-family model type: {modelType}')
    return archType


def permute(tensor, nHeads: int):
    return tensor.reshape(
        nHeads, 2, tensor.shape[0] // nHeads // 2, *tensor.shape[1:]
    ).swapaxes(1, 2).reshape(tensor.shape).contiguous()


class LlamaModelPlan(base.BaseModelPlan):
    headerMagic = HEADER_MAGIC
    headerKeys = HEADER_KEYS

    def __init__(self, folderPath, sourceConfig, weightsFloatType, weightMap):
        super().__init__(folderPath, sourceConfig, weightsFloatType, weightMap)
        self.archType = parseArchType(sourceConfig['model_type'])
        self.__buildHeader()
        self.__buildTensors()

    def __buildHeader(self):
        config = self.sourceConfig
        nExperts = int(config.get('num_experts', 0))
        self.header = {
            'version': 0,
            'arch_type': self.archType,
            'dim': int(config['hidden_size']),
            'hidden_dim': int(config['intermediate_size']),
            'n_layers': int(config['num_hidden_layers']),
            'n_heads': int(config['num_attention_heads']),
            'n_kv_heads': int(config['num_key_value_heads']),
            'n_experts': nExperts,
            'n_active_experts': int(config.get('num_experts_per_tok', 0)),
            'vocab_size': int(config['vocab_size']),
            'max_seq_len': int(config['max_position_embeddings']),
            'hidden_act': base.parseHiddenAct(config['hidden_act']),
            'weights_float_type': self.weightsFloatType,
        }

        if config.get('rope_theta') is not None:
            self.header['rope_theta'] = int(config['rope_theta'])
        ropeScaling = config.get('rope_scaling')
        if ropeScaling is not None:
            self.header['rope_scaling_factor'] = int(ropeScaling['factor'])
            self.header['rope_scaling_low_freq_factor'] = int(ropeScaling['low_freq_factor'])
            self.header['rope_scaling_high_freq_factory'] = int(ropeScaling['high_freq_factor'])
            self.header['rope_scaling_orig_max_seq_len'] = int(
                ropeScaling['original_max_position_embeddings']
            )
            self.header['rope_type'] = base.parseRopeType(ropeScaling['rope_type'])
        if config.get('head_dim') is not None:
            self.header['head_dim'] = int(config['head_dim'])
        if config.get('rms_norm_eps') is not None:
            self.header['norm_epsilon'] = base.parseRmsNormEpsilon(float(config['rms_norm_eps']))
        if config.get('moe_intermediate_size') is not None:
            self.header['moe_hidden_dim'] = int(config['moe_intermediate_size'])

    def __buildTensors(self):
        h = self.header
        wt = self.weightsFloatType
        dim = h['dim']
        headDim = h.get('head_dim', dim // h['n_heads'])
        qDim = h['n_heads'] * headDim
        kvDim = h['n_kv_heads'] * headDim
        isQwen = self.archType in (ArchType.QWEN3, ArchType.QWEN3_MOE)

        self.addTensor('embedding', base.FloatType.F32, 'model.embed_tokens.weight',
            (h['vocab_size'], dim))
        for layerIndex in range(h['n_layers']):
            layer = f'model.layers.{layerIndex}'
            qTransform = (lambda tensor, n=h['n_heads']: permute(tensor, n)) \
                if self.archType == ArchType.LLAMA else None
            kTransform = (lambda tensor, n=h['n_kv_heads']: permute(tensor, n)) \
                if self.archType == ArchType.LLAMA else None
            self.addTensor('block_matmul_q', wt, f'{layer}.self_attn.q_proj.weight',
                (qDim, dim), transform=qTransform)
            self.addTensor('block_matmul_k', wt, f'{layer}.self_attn.k_proj.weight',
                (kvDim, dim), transform=kTransform)
            self.addTensor('block_matmul_v', wt, f'{layer}.self_attn.v_proj.weight', (kvDim, dim))
            self.addTensor('block_matmul_wo', wt, f'{layer}.self_attn.o_proj.weight', (dim, qDim))

            if h['n_experts'] > 0:
                moeHiddenDim = h['moe_hidden_dim']
                self.addTensor('block_moe_gate', base.FloatType.F32, f'{layer}.mlp.gate.weight',
                    (h['n_experts'], dim))
                for expertIndex in range(h['n_experts']):
                    expert = f'{layer}.mlp.experts.{expertIndex}'
                    self.addTensor('block_matmul_w1', wt, f'{expert}.gate_proj.weight',
                        (moeHiddenDim, dim))
                    self.addTensor('block_matmul_w2', wt, f'{expert}.down_proj.weight',
                        (dim, moeHiddenDim))
                    self.addTensor('block_matmul_w3', wt, f'{expert}.up_proj.weight',
                        (moeHiddenDim, dim))
            else:
                hiddenDim = h['hidden_dim']
                self.addTensor('block_matmul_w1', wt, f'{layer}.mlp.gate_proj.weight',
                    (hiddenDim, dim))
                self.addTensor('block_matmul_w2', wt, f'{layer}.mlp.down_proj.weight',
                    (dim, hiddenDim))
                self.addTensor('block_matmul_w3', wt, f'{layer}.mlp.up_proj.weight',
                    (hiddenDim, dim))

            if isQwen:
                self.addTensor('block_norm_q', base.FloatType.F32,
                    f'{layer}.self_attn.q_norm.weight', (headDim,))
                self.addTensor('block_norm_k', base.FloatType.F32,
                    f'{layer}.self_attn.k_norm.weight', (headDim,))
            self.addTensor('block_norm_0', base.FloatType.F32,
                f'{layer}.input_layernorm.weight', (dim,))
            self.addTensor('block_norm_1', base.FloatType.F32,
                f'{layer}.post_attention_layernorm.weight', (dim,))

        self.addTensor('final_norm', base.FloatType.F32, 'model.norm.weight', (dim,))
        self.addTensor('final_matmul_logits', wt,
            ['lm_head.weight', 'model.embed_tokens.weight'], (h['vocab_size'], dim))


if __name__ == '__main__':
    base.runConverter(LlamaModelPlan, 'Llama-family Hugging Face', 'llama3_8b')
