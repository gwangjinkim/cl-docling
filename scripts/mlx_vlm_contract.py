"""Narrow M6.2 policy, independent of optional numerical packages."""
VERSION = '0.6.14'
SOURCE_REVISION = '625f71fae24f0d5c5ee7f1ec747094e815393405'
ATOL, RTOL = 1e-3, 3e-4
OUTPUT_FIELDS = ('prompt_ids', 'tiles', 'tokens', 'raw', 'stop_reason')


def validate_config(config):
    """Reject configurations outside this checkpoint-specific correction audit."""
    expected = dict(model_type='idefics3', image_token_id=49190, scale_factor=4)
    text = dict(hidden_size=576, num_hidden_layers=30, num_attention_heads=9,
                num_key_value_heads=3, rope_theta=100000, vocab_size=49280)
    vision = dict(hidden_size=768, image_size=512, patch_size=16, num_attention_heads=12)
    for actual, values in ((config, expected), (config['text_config'], text),
                           (config['vision_config'], vision)):
        if any(actual.get(key) != value for key, value in values.items()):
            raise ValueError('Unsupported M6.2 checkpoint configuration')
    v, t = config['vision_config'], config['text_config']
    if (v.get('hidden_act', 'gelu_pytorch_tanh') != 'gelu_pytorch_tanh'
            or v.get('layer_norm_eps', 1e-6) != 1e-6
            or v.get('num_hidden_layers', 12) != 12
            or t.get('tie_word_embeddings', False)
            or config.get('quantization') or config.get('quantization_config')):
        raise ValueError('Unsupported M6.2 checkpoint semantics')
    return 'gelu_pytorch_tanh', 1e-6


def compare_output(actual, expected):
    return [field for field in OUTPUT_FIELDS if actual[field] != expected[field]]
