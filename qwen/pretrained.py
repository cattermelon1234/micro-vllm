"""Load an official dense Qwen3 checkpoint into our implementation."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import QwenConfig
from qwen import Qwen

DEFAULT_MODEL = 'Qwen/Qwen3-0.6B'


def copy_reference_weights(ours, reference):
    weights = {}
    for name, tensor in reference.state_dict().items():
        name = name.removeprefix('model.')
        name = name.replace('.self_attn.', '.attention.')
        name = name.replace('.o_proj.', '.proj_out.')
        weights[name] = tensor
    if ours.config.tie_word_embeddings:
        weights['lm_head.weight'] = weights['embed_tokens.weight']
    ours.load_state_dict(weights, strict=True)


def load_pretrained(model_id=DEFAULT_MODEL, *, revision='main', dtype=torch.float32,
                    device='cpu', keep_reference=False):
    # Keep downloads inside the project; HF caches subsequent runs.
    options = dict(revision=revision, cache_dir='.cache/huggingface')
    reference = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, attn_implementation='eager',
        use_safetensors=True, **options,
    ).eval()
    config = QwenConfig.from_hf(reference.config)

    # create shapes only, no values
    with torch.device('meta'):
        ours = Qwen(config)
    ours = ours.to(dtype=dtype).to_empty(device='cpu') # to_empty because we are eventually loading in real weights

    if config.tie_word_embeddings:
        ours.lm_head.weight = ours.embed_tokens.weight
    copy_reference_weights(ours, reference)
    ours = ours.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left', **options)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ours.config.pad_token_id = tokenizer.pad_token_id
    if keep_reference:
        return ours, tokenizer, reference.to(device)
    return ours, tokenizer
