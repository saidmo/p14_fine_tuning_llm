"""Compare le meilleur checkpoint SFT local et l'adaptateur Hugging Face publie.

Dependances : huggingface_hub, safetensors, torch (facultatif pour .bin).
Ne charge pas le modele Qwen : verification sur CPU suffisante.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open


def find_state(path):
    options = [path / 'trainer_state.json', path.parent / 'trainer_state.json']
    options += sorted(path.glob('checkpoint-*/trainer_state.json'))
    options += sorted(path.parent.glob('checkpoint-*/trainer_state.json'))
    for p in options:
        if p.is_file():
            data = json.loads(p.read_text())
            if data.get('best_model_checkpoint'):
                return p, data
    return None, None


def read_weights(path):
    files = sorted(path.glob('adapter_model*.safetensors'))
    if files:
        result = {}
        for filename in files:
            with safe_open(str(filename), framework='pt', device='cpu') as reader:
                for key in reader.keys():
                    if key in result:
                        raise RuntimeError(f'Cle dupliquee: {key}')
                    result[key] = reader.get_tensor(key)
        return result, [str(x) for x in files]
    files = sorted(path.glob('adapter_model*.bin'))
    if files:
        import torch
        result = {}
        for filename in files:
            data = torch.load(filename, map_location='cpu', weights_only=True)
            result.update(data)
        return result, [str(x) for x in files]
    raise FileNotFoundError(f'Poids adaptateur introuvables sous {path}')


def comparison(a, b):
    import torch
    keys_a, keys_b = set(a), set(b)
    missing_local = sorted(keys_b - keys_a)
    missing_hub = sorted(keys_a - keys_b)
    different = []
    maximum = 0.0
    for key in sorted(keys_a & keys_b):
        x, y = a[key], b[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            different.append({'tensor': key, 'problem': f'shape/dtype: {tuple(x.shape)} {x.dtype} vs {tuple(y.shape)} {y.dtype}'})
            continue
        if not torch.equal(x, y):
            gap = (x.float() - y.float()).abs().max().item()
            maximum = max(maximum, gap)
            different.append({'tensor': key, 'max_abs_diff': gap})
    return {'local_tensors': len(a), 'hub_tensors': len(b),
            'missing_local': missing_local, 'missing_hub': missing_hub,
            'different_count': len(different), 'max_abs_diff': maximum,
            'different_examples': different[:10],
            'exact_match': not missing_local and not missing_hub and not different}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True, help='Repertoire du run SFT (/workspace/outputs/sft/sft-...)')
    p.add_argument('--adapter', required=True, help='Depot HF prive de l adaptateur')
    p.add_argument('--revision', default=None, help='Commit SHA du depot HF (recommande)')
    p.add_argument('--checkpoint', default=None, help='Chemin vers checkpoint-550, si connu')
    args = p.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if not output.is_dir():
        p.error(f'Repertoire introuvable: {output}')
    state_path, state = find_state(output)
    if not state:
        p.error('trainer_state.json avec best_model_checkpoint introuvable. Fournir un repertoire de run contenant les checkpoints.')
    best = Path(state['best_model_checkpoint'])
    if not best.is_dir():
        best = output / Path(state['best_model_checkpoint']).name
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else best
    result = {'trainer_state': str(state_path), 'best_model_checkpoint_reported': state['best_model_checkpoint'],
              'best_metric': state.get('best_metric'), 'global_step': state.get('global_step'),
              'checkpoint_compared': str(checkpoint)}
    if not checkpoint.is_dir():
        print(json.dumps(result, indent=2, ensure_ascii=False))
        p.error(f'Meilleur checkpoint non trouve: {checkpoint}. Impossible de prouver identite des poids sans lui.')
    api = HfApi()
    info = api.model_info(args.adapter, revision=args.revision)
    result['hub_commit_sha'] = info.sha
    repo = Path(snapshot_download(repo_id=args.adapter, revision=info.sha,
                                  allow_patterns=['adapter_model*.safetensors', 'adapter_model*.bin', 'adapter_config.json']))
    a, local_files = read_weights(checkpoint)
    b, hub_files = read_weights(repo)
    result['local_weight_files'] = local_files
    result['hub_weight_files'] = [str(x) for x in hub_files]
    result['comparison'] = comparison(a, b)
    for name, location in [('local', checkpoint), ('hub', repo)]:
        conf = location / 'adapter_config.json'
        if conf.is_file():
            cfg = json.loads(conf.read_text())
            result[f'{name}_config'] = {
                'base_model_name_or_path': cfg.get('base_model_name_or_path'),
                'trainable_token_indices': cfg.get('trainable_token_indices'),
                'r': cfg.get('r'), 'lora_alpha': cfg.get('lora_alpha')}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result['comparison']['exact_match']:
        raise SystemExit('FAIL: poids du checkpoint et du Hub differents (ou format de cles different).')
    print('PASS: poids Hub STRICTEMENT IDENTIQUES au meilleur checkpoint retenu.')
    print('Attention : ce test ne prouve pas que le depot Hub provient du run MLflow attendu ; verifier dates, revisions et tags.')


if __name__ == '__main__':
    main()
