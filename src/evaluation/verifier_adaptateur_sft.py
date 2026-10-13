"""Audit numerique du chargement PEFT et de merge_and_unload (GPU recommande).

Exemple:
python verifier_adaptateur_sft.py --base Qwen/Qwen3-1.7B-Base --adapter saidmo2/chsa-qwen3-1.7b-sft-lora
"""
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

TOKENS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]


def snapshots(model, ids):
    model.eval()
    with torch.no_grad():
        embeddings = model.get_input_embeddings()
        device = next(embeddings.parameters()).device
        inp = torch.tensor(ids, device=device)
        vecs_in = embeddings(inp).detach().float().cpu()
        # lm_head may be wrapped by PEFT. Testing outputs directly avoids bypassing wrappers.
        d = model.get_output_embeddings()
        hidden = torch.eye(model.config.hidden_size, dtype=next(d.parameters()).dtype, device=next(d.parameters()).device)[:4]
        # Instead of storing giant logits, only sample selected output columns.
        logits = d(hidden)[:, ids].detach().float().cpu()
    return vecs_in, logits


def distances(a, b, names):
    return {k: round(float(torch.linalg.vector_norm(a[i] - b[i])), 8) for i, k in enumerate(names)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='Qwen/Qwen3-1.7B-Base')
    p.add_argument('--adapter', required=True, help='Hub repo ID or local adapter path')
    p.add_argument('--revision', default=None, help='Pin adapter SHA when possible')
    p.add_argument('--base-revision', default=None)
    p.add_argument('--device-map', default='auto')
    args = p.parse_args()
    cfg = PeftConfig.from_pretrained(args.adapter, revision=args.revision)
    print('Adapter config base_model_name_or_path:', cfg.base_model_name_or_path)
    print('Adapter trainable_token_indices:', cfg.trainable_token_indices)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, revision=args.revision)
    token_ids = tokenizer.convert_tokens_to_ids(TOKENS)
    assert all(isinstance(i, int) and i != tokenizer.unk_token_id for i in token_ids), token_ids
    assert len(set(token_ids)) == len(token_ids), token_ids
    print('Tokens:', dict(zip(TOKENS, token_ids)))
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base, revision=args.base_revision,
                                                torch_dtype=dtype, device_map=args.device_map)
    before_in, before_out = snapshots(model, token_ids)
    peft_model = PeftModel.from_pretrained(model, args.adapter, revision=args.revision)
    adapted_in, adapted_out = snapshots(peft_model, token_ids)
    merged = peft_model.merge_and_unload()
    merged_in, merged_out = snapshots(merged, token_ids)
    results = {
        'token_ids': dict(zip(TOKENS, token_ids)),
        'input_embedding_delta_base_to_peft': distances(before_in, adapted_in, TOKENS),
        'input_embedding_delta_peft_to_merged': distances(adapted_in, merged_in, TOKENS),
        'output_projection_max_abs_peft_to_merged': float((adapted_out-merged_out).abs().max()),
        'embedding_weight_tied_merged': merged.get_input_embeddings().weight.data_ptr() == merged.get_output_embeddings().weight.data_ptr(),
        'dtype': str(dtype),
    }
    print(json.dumps(results, indent=2, ensure_ascii=False))
    assert all(v > 0 for v in results['input_embedding_delta_base_to_peft'].values()), 'Au moins un token reste inchange: verifier ce qui devait etre appris'
    assert max(results['input_embedding_delta_peft_to_merged'].values()) < 0.01, 'Embeddings modifies durant fusion'
    assert results['output_projection_max_abs_peft_to_merged'] < 0.02, 'Sortie differente apres fusion'
    print('PASS: audit des vecteurs avant / apres merge. Tester aussi la generation sur plusieurs prompts.')


if __name__ == '__main__':
    main()
