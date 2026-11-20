"""R1-7a plug task (gen_demo.py HDF5, 11-19), adapted from train_smolvla_clean50.py. Fine-tune the official SmolVLA on pinned RoboTwin clean50 HDF5 with DDP.

JPEGs follow RoboTwin's cv2-imdecode-as-RGB convention. State v[t] predicts
v[t+1:t+51]; padded timesteps and unused action dimensions do not enter loss.
Checkpoints contain standard LeRobot policy/processors plus optimizer/RNG state.
"""
import argparse
import contextlib
import functools
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import cv2
import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Updates with a non-finite gradient norm are skipped and logged; abort after this many.
MAX_NONFINITE_SKIPS = 20


def ok_episodes(config):
    """First config['episodes'] successful demos (gen_demo.py meta.json ok=true), ordered by seed."""
    root = Path(config['data_dir'])
    eps = sorted(p.parent for p in root.glob('s*/meta.json') if json.loads(p.read_text()).get('ok') and (p.parent / 'episode.hdf5').exists())
    return eps[:config['episodes']]


def prepare(config, smoke=False):
    root = Path(config['root'])
    eps = ok_episodes(config)[:1 if smoke else None]
    assert smoke or len(eps) == config['episodes'], (len(eps), config['episodes'])
    records, states, actions = [], [], []
    for d in eps:
        path = d / 'episode.hdf5'
        with h5py.File(path, 'r') as h:
            v = np.asarray(h['joint_action/vector'], dtype=np.float32)
            q = np.asarray(h['observation/joint_state'], dtype=np.float32)
            assert v.ndim == 2 and v.shape[1] == config['action_dim'] and q.shape == v.shape and np.isfinite(v).all(), path
            for camera in config['camera_mapping'].values():
                assert len(h[f'observation/{camera}/rgb']) == len(v), (path, camera)
            instruction = h.attrs['instruction']
        records.append({'path': str(path), 'episode': d.name, 'frames': len(v), 'instructions': [instruction]})
        states.append(q)
        actions.append(v[np.minimum(np.arange(len(v)) + 1, len(v) - 1)])
    stats = {}
    for key, arrays in [('observation.state', states), ('action', actions)]:
        values = np.concatenate(arrays).astype(np.float64)
        stats[key] = {name: fn(values, axis=0).tolist() for name, fn in
                      [('mean', np.mean), ('std', np.std), ('min', np.min), ('max', np.max)]}
        stats[key]['count'] = [len(values)]
    payload = {'data_dir': config['data_dir'], 'episode_count': len(records), 'task_count': 1,
               'num_frames': sum(r['frames'] for r in records), 'randomized_data_used': True,
               'state_definition': 'measured joints J1..J7 (rad) + gripper opening (0 closed..1 open)',
               'action_definition': 'commanded targets cmd[min(t+1+k,L-1)], k=0..49 (same 8 dims)',
               'color_convention': 'JPEGs are RGB (PIL); cv2.imdecode BGR is converted to RGB',
               'stats': stats, 'episodes': records}
    target = root / config['index_file']
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(target)
    print('DATASET_READY', json.dumps({k: v for k, v in payload.items() if k not in ('stats', 'episodes')}), flush=True)
    return target


class Clean50(Dataset):
    def __init__(self, config, index):
        self.config, self.root = config, Path(config['root'])
        self.records = index['episodes']
        self.ends = np.cumsum([r['frames'] for r in self.records])

    def __len__(self):
        return int(self.ends[-1])

    @functools.lru_cache(maxsize=128)
    def episode(self, idx):
        h = h5py.File(self.records[idx]['path'], 'r')
        return h, np.asarray(h['joint_action/vector'], dtype=np.float32), np.asarray(h['observation/joint_state'], dtype=np.float32)

    def locate(self, idx):
        episode = int(np.searchsorted(self.ends, idx, side='right'))
        return episode, int(idx - (self.ends[episode - 1] if episode else 0))

    def __getitem__(self, idx):
        episode, frame = self.locate(idx)
        record = self.records[episode]
        h, v, q = self.episode(episode)
        future = frame + 1 + np.arange(self.config['chunk_size'])
        sample = {'observation.state': torch.from_numpy(q[frame].copy()),
                  'action': torch.from_numpy(v[np.minimum(future, len(v) - 1)].copy()),
                  'action_is_pad': torch.from_numpy(future >= len(v)),
                  'task': record['instructions'][idx % len(record['instructions'])]}
        height, width = self.config['image_size']
        for key, camera in self.config['camera_mapping'].items():
            bits = h[f'observation/{camera}/rgb'][frame]
            rgb = cv2.imdecode(np.frombuffer(bits.tobytes(), np.uint8), cv2.IMREAD_COLOR)
            if rgb is not None: rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            if rgb is None:
                raise ValueError(f'Invalid JPEG: {record["path"]} frame={frame} camera={camera}')
            if rgb.shape[:2] != (height, width):
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
            sample[key] = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255)
        return sample


def worker_init(_):
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state(state['cuda'].cpu())


class ResumableDistributedSampler(DistributedSampler):
    """Skip consumed indices without decoding their images again."""
    start_index = 0

    def __iter__(self):
        return itertools.islice(super().__iter__(), self.start_index, None)

    def __len__(self):
        return super().__len__() - self.start_index


def lr_factor(config, step):
    start = config.get('continuation_start_step')
    if start is not None:
        local_step = max(0, step - start)
        initial = config['continuation_initial_learning_rate'] / config['learning_rate']
        if local_step < config['warmup_steps']:
            return initial + (1 - initial) * local_step / config['warmup_steps']
        fraction = min(1., (local_step - config['warmup_steps']) /
                       (config['steps'] - start - config['warmup_steps']))
    else:
        if step < config['warmup_steps']:
            return (step + 1) / config['warmup_steps']
        fraction = min(1., (step - config['warmup_steps']) /
                       (config['steps'] - config['warmup_steps']))
    floor = config['min_learning_rate'] / config['learning_rate']
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * fraction))


def validate_resume_config(config, saved_config, saved_step):
    if config == saved_config:
        return False
    allowed = {'steps', 'output_dir', 'learning_rate', 'warmup_steps',
               'min_learning_rate', 'continuation_start_step',
               'continuation_initial_learning_rate'}
    changed = {key for key in config.keys() | saved_config.keys()
               if config.get(key) != saved_config.get(key)}
    assert not changed - allowed, f'Resume changes training/data settings: {sorted(changed - allowed)}'
    assert config.get('continuation_start_step') == saved_step, 'Continuation must start at the saved step.'
    assert config['steps'] > saved_step + config['warmup_steps'], 'Continuation needs warmup and decay steps.'
    assert config['output_dir'] != saved_config['output_dir'], 'Continuation needs a separate output directory.'
    assert 0 < config['continuation_initial_learning_rate'] <= config['learning_rate']
    assert 0 < config['min_learning_rate'] <= config['learning_rate']
    assert config['warmup_steps'] > 0
    return True


def train(config, resume=None):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    class RobotwinPolicy(SmolVLAPolicy):
        def __init__(self, policy_config):
            super().__init__(policy_config)
            # Convert before loading so FP32 resume weights are not rounded to BF16.
            self.float()

        def forward(self, batch):
            images, masks = self.prepare_images(batch)
            losses = self.model(images, masks, batch[OBS_LANGUAGE_TOKENS],
                                batch[OBS_LANGUAGE_ATTENTION_MASK], self.prepare_state(batch),
                                self.prepare_action(batch))[:, :, :config['action_dim']]
            valid = (~batch['action_is_pad']).to(losses.dtype).unsqueeze(-1)
            return (losses * valid).sum() / (valid.sum() * config['action_dim']).clamp_min(1)

    rank, local_rank, world = (int(os.environ.get(k, default)) for k, default in
                               [('RANK', 0), ('LOCAL_RANK', 0), ('WORLD_SIZE', 1)])
    assert world == config['world_size'], (world, config['world_size'])
    torch.cuda.set_device(local_rank)
    if world > 1:
        from datetime import timedelta
        dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    device = torch.device('cuda', local_rank)
    random.seed(config['seed'] + rank)
    np.random.seed(config['seed'] + rank)
    torch.manual_seed(config['seed'] + rank)
    cv2.setNumThreads(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    root, output = Path(config['root']), Path(config['root']) / config['output_dir']
    index_path = root / config['index_file']
    index = json.loads(index_path.read_text())
    assert index['episode_count'] == (1 if config.get('smoke_test') else config['episodes'])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
        shutil.copy2(index_path, output / 'dataset_provenance.json')
        shutil.copy2(__file__, output / 'training_source.py')
    base = Path(resume) / 'pretrained_model' if resume else root / config['pretrained']
    policy_config = PreTrainedConfig.from_pretrained(base, local_files_only=True)
    policy_config.device = str(device)
    policy_config.vlm_model_name = str(root / config['vlm'])
    policy_config.load_vlm_weights = False
    policy_config.freeze_vision_encoder = config['freeze_vision_encoder']
    policy_config.train_expert_only = config['train_expert_only']
    policy_config.push_to_hub = False
    policy_config.input_features = {'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(config['action_dim'],))}
    for key in config['camera_mapping']:
        policy_config.input_features[key] = PolicyFeature(type=FeatureType.VISUAL,
                                                           shape=(3, *config['image_size']))
    policy_config.output_features = {'action': PolicyFeature(type=FeatureType.ACTION, shape=(config['action_dim'],))}
    policy = RobotwinPolicy.from_pretrained(base, config=policy_config, local_files_only=True, strict=True)
    policy.train()
    # Keep FP32 weights/optimizer states and use BF16 only in forward/backward.
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config['learning_rate'], betas=(0.9, 0.95),
                                 eps=1e-8, weight_decay=config['weight_decay'], foreach=False)
    schedule = functools.partial(lr_factor, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    stats = {key: {k: torch.tensor(v, dtype=torch.float32) for k, v in value.items()}
             for key, value in index['stats'].items()}
    preprocessor, postprocessor = make_smolvla_pre_post_processors(policy_config, stats)
    dataset = Clean50(config, index)
    sampler = ResumableDistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                          seed=config['seed'], drop_last=True)
    loader = DataLoader(dataset, batch_size=config['batch_size_per_gpu'], sampler=sampler,
                        num_workers=config['num_workers'], pin_memory=True, drop_last=True,
                        persistent_workers=config['num_workers'] > 0, worker_init_fn=worker_init,
                        generator=torch.Generator().manual_seed(config['seed'] + rank))
    step, epoch, offset = 0, 0, 0
    assert resume or config.get('continuation_start_step') is None, 'Continuation requires a checkpoint.'
    if resume:
        state = torch.load(Path(resume) / 'training_state.pt', map_location='cpu', weights_only=False)
        continuation = validate_resume_config(config, state['config'], state['step'])
        assert index == json.loads((Path(resume) / 'dataset_provenance.json').read_text()), 'Resume dataset changed.'
        optimizer.load_state_dict(state['optimizer'])
        step, epoch, offset = state['step'], state['epoch'], state['batch_offset']
        if continuation:
            for group in optimizer.param_groups:
                assert math.isclose(group['lr'], config['continuation_initial_learning_rate']), 'Initial LR must match checkpoint.'
                group['initial_lr'] = config['learning_rate']
                group['lr'] = config['learning_rate']
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule, last_epoch=step - 1)
        else:
            scheduler.load_state_dict(state['scheduler'])
        assert scheduler.last_epoch == step, (scheduler.last_epoch, step)
        for parameter, optimizer_state in optimizer.state.items():
            for key in ('exp_avg', 'exp_avg_sq'):
                assert optimizer_state[key].device == parameter.device
                assert optimizer_state[key].dtype == torch.float32
        restore_rng(state['rng_states'][rank])
        if rank == 0:
            from safetensors import safe_open
            checked = []
            with safe_open(base / 'model.safetensors', framework='pt', device='cpu') as saved:
                for tag in ('.vision_model.', '.text_model.layers.', '.lm_expert.layers.'):
                    name, parameter = next((n, p) for n, p in policy.named_parameters() if tag in n)
                    assert torch.equal(parameter.detach().cpu(), saved.get_tensor(name)), name
                    checked.append(name)
            print('RESUME_VALIDATED', json.dumps({'step': step, 'epoch': epoch, 'batch_offset': offset,
                  'optimizer_dtype': 'float32', 'exact_parameter_checks': checked,
                  'optimizer_states_restored': len(optimizer.state),
                  'learning_rate': optimizer.param_groups[0]['lr'],
                  'schedule_restarted': continuation}), flush=True)
            if continuation:
                provenance = {'source_checkpoint': str(Path(resume).resolve()), 'source_step': step,
                              'source_config': state['config'], 'continuation_config': config,
                              'optimizer_rng_and_dataset_position_restored': True}
                (output / 'continuation_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        del state
    model = DistributedDataParallel(policy, device_ids=[local_rank], find_unused_parameters=True,
                                     gradient_as_bucket_view=True, bucket_cap_mb=config.get('ddp_bucket_cap_mb', 25)) if world > 1 else policy
    # Each rank checks its own accumulated gradients before the all-reduce, so a non-finite
    # global norm can be traced to the rank (GPU) and samples that produced it.
    local_nonfinite = torch.zeros((), dtype=torch.bool, device=device)
    if world > 1:
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

        def finite_check_hook(flag, bucket):
            flag.logical_or_(~torch.isfinite(bucket.buffer()).all())
            return default_hooks.allreduce_hook(None, bucket)
        model.register_comm_hook(local_nonfinite, finite_check_hook)
    skipped = 0
    sampler.set_epoch(epoch)
    assert 0 <= offset <= sampler.num_samples // config['batch_size_per_gpu']
    sampler.start_index = offset * config['batch_size_per_gpu']
    iterator = iter(loader)
    initial_step = step
    if rank == 0:
        print('TRAINING_READY', json.dumps({'world_size': world, 'frames': len(dataset),
              'episodes': index['episode_count'], 'trainable_params': sum(p.numel() for p in params),
              'total_params': sum(p.numel() for p in policy.parameters()),
              'global_batch_size': config['batch_size_per_gpu'] * world * config['gradient_accumulation_steps'],
              'freeze_vision_encoder': policy_config.freeze_vision_encoder,
              'train_expert_only': policy_config.train_expert_only,
              'parameter_dtypes': sorted({str(p.dtype) for p in params}),
              'steps': config['steps'], 'output': str(output)}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    log_loss = torch.zeros((), device=device)
    log_updates = 0
    def save_checkpoint():
        rngs = [None] * world
        if world > 1:
            dist.all_gather_object(rngs, rng_state())
        else:
            rngs[0] = rng_state()
        if rank == 0:
            checkpoint = output / 'checkpoints' / f'step_{step:06d}'
            temp = checkpoint.with_name(checkpoint.name + '.tmp')
            temp.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(temp / 'pretrained_model')
            preprocessor.save_pretrained(temp / 'pretrained_model')
            postprocessor.save_pretrained(temp / 'pretrained_model')
            torch.save({'step': step, 'epoch': epoch, 'batch_offset': offset,
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'rng_states': rngs, 'config': config}, temp / 'training_state.pt')
            shutil.copy2(index_path, temp / 'dataset_provenance.json')
            temp.rename(checkpoint)
            latest = output / 'checkpoints/last'
            latest.unlink(missing_ok=True)
            latest.symlink_to(checkpoint.name, target_is_directory=True)
            print('CHECKPOINT', str(checkpoint), flush=True)
        if world > 1:
            dist.barrier()
    while step < config['steps']:
        local_nonfinite.zero_()
        step_loss = torch.zeros((), device=device)
        for micro in range(config['gradient_accumulation_steps']):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch, offset = epoch + 1, 0
                sampler.start_index = 0
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            offset += 1
            batch = preprocessor(batch)
            sync = micro == config['gradient_accumulation_steps'] - 1
            context = model.no_sync() if world > 1 and not sync else contextlib.nullcontext()
            with context:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss rank={rank} step={step}')
                (loss / config['gradient_accumulation_steps']).backward()   # outside autocast: 2x faster on 11-03 tests
            step_loss += loss.detach() / config['gradient_accumulation_steps']
        grad_norm = torch.nn.utils.clip_grad_norm_(params, config['grad_clip_norm'])
        if not torch.isfinite(grad_norm):
            skipped += 1
            # The micro-batches of this update, from the sampler's deterministic per-epoch order.
            batch_size = config['batch_size_per_gpu']
            order = list(DistributedSampler.__iter__(sampler))
            ids = order[max(0, (offset - config['gradient_accumulation_steps']) * batch_size):offset * batch_size]
            local = {'rank': rank, 'local_nonfinite': bool(local_nonfinite),
                     'samples': [[dataset.records[e]['path'], f] for e, f in map(dataset.locate, ids)]}
            reports = [None] * world
            if world > 1:
                dist.all_gather_object(reports, local)
            else:
                reports[0] = local
            if rank == 0:
                record = {'step': step, 'epoch': epoch, 'skipped_total': skipped, 'ranks': reports}
                print('NONFINITE_GRAD', json.dumps(record), flush=True)
                with (output / 'nonfinite.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
            assert skipped <= MAX_NONFINITE_SKIPS, f'Too many non-finite gradient updates: {skipped}'
            optimizer.zero_grad(set_to_none=True)
            continue
        log_loss += step_loss
        if step == initial_step and rank == 0:
            groups = {}
            for group, tag in [('vision', '.vision_model.'), ('language', '.text_model.layers.'),
                               ('action_expert', '.lm_expert.layers.')]:
                for name, parameter in policy.named_parameters():
                    if tag in name and parameter.grad is not None:
                        groups[group] = {'parameter': name, 'trainable': parameter.requires_grad,
                                         'grad_abs_mean': parameter.grad.float().abs().mean().item()}
                        break
                assert group in groups and groups[group]['trainable'], f'Missing gradients in {group}'
            print('FULL_FINETUNE_GRADIENTS', json.dumps(groups), flush=True)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step, log_updates = step + 1, log_updates + 1
        if step == 1 or step % config['log_freq'] == 0:
            if world > 1:
                dist.all_reduce(log_loss)
            if rank == 0:
                record = {'step': step, 'loss': log_loss.item() / (world * log_updates),
                          'grad_norm': float(grad_norm), 'lr': optimizer.param_groups[0]['lr'],
                          'seconds_per_step': (time.perf_counter() - started) / log_updates,
                          'epoch': epoch, 'gpu_peak_gb': torch.cuda.max_memory_allocated() / 1e9}
                print('METRIC', json.dumps(record), flush=True)
                with (output / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
            log_loss.zero_()
            log_updates, started = 0, time.perf_counter()
        # Early checkpoint verifies the save/resume path before a long run.
        if step == 10 or step % config['save_freq'] == 0 or step == config['steps']:
            save_checkpoint()
    if rank == 0:
        (output / 'COMPLETE').write_text(f'step={step}\n')
        print('TRAINING_COMPLETE', step, flush=True)
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.smoke_test:
        config.update(smoke_test=True, steps=int(os.environ.get('SMOKE_STEPS', 10)), warmup_steps=2, num_workers=4,
                      index_file=config['index_file'].replace('.json', '_smoke.json'),
                      output_dir=config['output_dir'] + '_smoke')
        if int(os.environ.get('RANK', '0')) == 0:
            prepare(config, smoke=True)
        # All ranks use the same small, explicitly labeled diagnostic dataset.
        while not (Path(config['root']) / config['index_file']).is_file():
            time.sleep(.2)
    if args.prepare_only:
        prepare(config)
    else:
        if not args.smoke_test and not (Path(config['root']) / config['index_file']).is_file():
            if int(os.environ.get('RANK', '0')) == 0: prepare(config)
            while not (Path(config['root']) / config['index_file']).is_file(): time.sleep(.5)
        train(config, args.resume)


if __name__ == '__main__':
    main()
