import hydra
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import gym
import pathlib
import time
import collections
import random
import json
import uuid
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import set_seed, parse_cfg, Logger
from cleandiffuser.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
from cleandiffuser.env.async_vector_env import AsyncVectorEnv
from cleandiffuser.env.utils import VideoRecorder
from cleandiffuser.dataset.robomimic_dataset import RobomimicImageDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters


# --------------------- Checkpoint helpers --------------------- #

def _get_rng_state(device: str):
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict, device: str):
    if not state:
        return
    try:
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        if torch.cuda.is_available() and str(device).startswith("cuda") and "torch_cuda_random_state_all" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda_random_state_all"])
    except Exception as e:
        print(f"[WARN] Failed to fully restore RNG state: {e}")


def save_full_checkpoint(
    ckpt_path: pathlib.Path,
    agent,
    optimizer,
    scheduler,
    step: int,
    device: str,
):
    """
    Full training checkpoint for true resume.
    """
    payload = {
        "step": int(step),
        "model": agent.model.state_dict(),
        "model_ema": agent.model_ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": _get_rng_state(device),
    }
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt_path)
    print(f"[CKPT] Saved full checkpoint: {ckpt_path}")


def load_full_checkpoint(
    ckpt_path: str,
    agent,
    optimizer,
    scheduler,
    device: str,
):
    """
    Returns: start_step (int)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    agent.model.load_state_dict(ckpt["model"])
    agent.model_ema.load_state_dict(ckpt["model_ema"])
    optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and ckpt.get("scheduler", None) is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    _set_rng_state(ckpt.get("rng", None), device=device)

    start_step = int(ckpt.get("step", 0))
    print(f"[CKPT] Loaded full checkpoint: {ckpt_path} (step={start_step})")
    return start_step


def save_inference_agent_pt(model_path: pathlib.Path, agent):
    """
    Maintain your existing inference pipeline: save ONLY {'model','model_ema'}.
    """
    payload = {
        "model": agent.model.state_dict(),
        "model_ema": agent.model_ema.state_dict(),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, model_path)
    print(f"[MODEL] Saved inference agent pt: {model_path}")


# --------------------- Env creation (unchanged) --------------------- #

def make_async_envs(args):
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    print(f"Starting to create {args.num_envs} asynchronous Robomimic environments...")

    def create_robomimic_env(env_meta, shape_meta, enable_render=True):
        modality_mapping = collections.defaultdict(list)
        for key, attr in shape_meta['obs'].items():
            modality_mapping[attr.get('type', 'low_dim')].append(key)
        ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=enable_render,
            use_image_obs=enable_render,
        )
        return env

    dataset_path = os.path.expanduser(args.dataset_path)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta['env_kwargs']['use_object_obs'] = False

    abs_action = args.abs_action
    if abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False

    def env_fn():
        env = create_robomimic_env(env_meta=env_meta, shape_meta=args.shape_meta)
        env.env.hard_reset = False
        return MultiStepWrapper(
            VideoRecordingWrapper(
                RobomimicImageWrapper(
                    env=env,
                    shape_meta=args.shape_meta,
                    init_state=None,
                    render_obs_key=args.render_obs_key
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=10,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=22,
                    thread_type='FRAME',
                    thread_count=1
                ),
                file_path=None,
                steps_per_render=2
            ),
            n_obs_steps=args.obs_steps,
            n_action_steps=args.action_steps,
            max_episode_steps=args.max_episode_steps
        )

    def dummy_env_fn():
        env = create_robomimic_env(env_meta=env_meta, shape_meta=args.shape_meta, enable_render=False)
        return MultiStepWrapper(
            VideoRecordingWrapper(
                RobomimicImageWrapper(
                    env=env,
                    shape_meta=args.shape_meta,
                    init_state=None,
                    render_obs_key=args.render_obs_key
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=10,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=22,
                    thread_type='FRAME',
                    thread_count=1
                ),
                file_path=None,
                steps_per_render=2
            ),
            n_obs_steps=args.obs_steps,
            n_action_steps=args.action_steps,
            max_episode_steps=args.max_episode_steps
        )

    env_fns = [env_fn] * args.num_envs
    envs = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
    envs.seed(args.seed)
    return envs


# --------------------- Inference (unchanged logic) --------------------- #

def add_lowdim_obs_noise(obs_dict, shape_meta, noise_std=0.0, noise_key_to_std=None):
    """
    Add Gaussian noise to low-dim observation tensors only.
    Assumes obs_dict values are already normalized tensors.
    Does NOT touch image keys.
    """
    if noise_std <= 0.0 and (not noise_key_to_std):
        return obs_dict

    noisy_obs = dict(obs_dict)

    for key, tensor in obs_dict.items():
        obs_attr = shape_meta["obs"].get(key, {})
        obs_type = obs_attr.get("type", "low_dim")
        if obs_type != "low_dim":
            continue

        key_std = noise_std
        if noise_key_to_std is not None and key in noise_key_to_std:
            key_std = float(noise_key_to_std[key])

        if key_std > 0.0:
            noisy_obs[key] = tensor + torch.randn_like(tensor) * key_std

    return noisy_obs

def inference(args, envs, dataset, agent, logger):
    episode_rewards = []
    episode_steps = []
    episode_success = []

    if args.diffusion == "ddpm":
        solver = None
    elif args.diffusion == "ddim":
        solver = "ddim"
    elif args.diffusion == "dpm":
        solver = "ode_dpmpp_2"
    elif args.diffusion == "edm":
        solver = "euler"
    else:
        solver = None

    for i in range(args.eval_episodes // args.num_envs):
        ep_reward = [0.0] * args.num_envs
        obs, t = envs.reset(), 0

        while t < args.max_episode_steps:
            obs_dict = {}
            for k in obs.keys():
                obs_seq = obs[k].astype(np.float32)
                nobs = dataset.normalizer['obs'][k].normalize(obs_seq)
                obs_dict[k] = torch.tensor(nobs, device=args.device, dtype=torch.float32)

            with torch.no_grad():
                prior = torch.zeros((args.num_envs, args.horizon, args.action_dim), device=args.device)
                naction, _ = agent.sample(
                    prior=prior,
                    n_samples=args.num_envs,
                    sample_steps=args.sample_steps,
                    solver=solver,
                    condition_cfg=obs_dict,
                    w_cfg=1.0,
                    temperature=args.temperature,
                    use_ema=True
                )

            naction = naction.detach().to('cpu').numpy()
            action_pred = dataset.normalizer['action'].unnormalize(naction)

            start = args.obs_steps - 1
            end = start + args.action_steps
            action = action_pred[:, start:end, :]

            if args.abs_action:
                action = dataset.undo_transform_action(action)

            obs, reward, done, info = envs.step(action)
            ep_reward += reward
            t += args.action_steps

        success = [1.0 if s > 0 else 0.0 for s in ep_reward]
        print(f"[Episode {1+i*(args.num_envs)}-{(i+1)*(args.num_envs)}] reward: {np.around(ep_reward, 2)} success:{success}")
        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)

    print(f"Mean step: {np.nanmean(episode_steps)} Mean reward: {np.nanmean(episode_rewards)} Mean success: {np.nanmean(episode_success)}")
    return {'mean_step': np.nanmean(episode_steps), 'mean_reward': np.nanmean(episode_rewards), 'mean_success': np.nanmean(episode_success)}


# --------------------- Pipeline --------------------- #

@hydra.main(config_path="../configs/dp/robomimic_multi_modal/chi_unet", config_name="lift_abs")
def pipeline(args):
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)

    dataset_path = os.path.expanduser(args.dataset_path)
    dataset = RobomimicImageDataset(
        dataset_path,
        horizon=args.horizon,
        shape_meta=args.shape_meta,
        n_obs_steps=args.obs_steps,
        pad_before=args.obs_steps - 1,
        pad_after=args.action_steps - 1,
        abs_action=args.abs_action
    )
    print(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

    # ---- Model creation (same as yours) ----
    if args.nn == "dit":
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import DiT1d
        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model,
            resize_shape=args.resize_shape, crop_shape=args.crop_shape, random_crop=args.random_crop,
            use_group_norm=args.use_group_norm, use_seq=args.use_seq
        ).to(args.device)
        nn_diffusion = DiT1d(
            args.action_dim, emb_dim=256 * args.obs_steps, d_model=320, n_heads=10,
            depth=2, timestep_emb_type="fourier"
        ).to(args.device)

    elif args.nn == "chi_unet":
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import ChiUNet1d
        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model,
            resize_shape=args.resize_shape, crop_shape=args.crop_shape, random_crop=args.random_crop,
            use_group_norm=args.use_group_norm, use_seq=args.use_seq
        ).to(args.device)
        nn_diffusion = ChiUNet1d(
            args.action_dim, 256, args.obs_steps, model_dim=256, emb_dim=256, dim_mult=[1, 2, 2],
            obs_as_global_cond=True, timestep_emb_type="positional"
        ).to(args.device)

    elif args.nn == "chi_transformer":
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import ChiTransformer
        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model,
            resize_shape=args.resize_shape, crop_shape=args.crop_shape, random_crop=args.random_crop,
            use_group_norm=args.use_group_norm, use_seq=args.use_seq, keep_horizon_dims=True
        ).to(args.device)
        nn_diffusion = ChiTransformer(
            args.action_dim, 256, args.horizon, args.obs_steps, d_model=256, nhead=4, num_layers=4,
            timestep_emb_type="positional"
        ).to(args.device)

    else:
        raise ValueError(f"Invalid nn type {args.nn}")

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================")
    print(f"======================= Parameter Report of Condition Model =======================")
    report_parameters(nn_condition)
    print(f"===================================================================================")

    x_max = torch.ones((1, args.horizon, args.action_dim), device=args.device) * +1.0
    x_min = torch.ones((1, args.horizon, args.action_dim), device=args.device) * -1.0

    if args.diffusion == "ddpm":
        from cleandiffuser.diffusion.ddpm import DDPM
        agent = DDPM(
            nn_diffusion=nn_diffusion,
            nn_condition=nn_condition,
            device=args.device,
            diffusion_steps=args.sample_steps,
            x_max=x_max,
            x_min=x_min,
            optim_params={"lr": args.lr}
        )
    elif args.diffusion == "edm":
        from cleandiffuser.diffusion.edm import EDM
        agent = EDM(
            nn_diffusion=nn_diffusion,
            nn_condition=nn_condition,
            device=args.device,
            optim_params={"lr": args.lr}
        )
    else:
        raise NotImplementedError

    optimizer = agent.optimizer
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.gradient_steps)

    # ------------------ Resume logic ------------------
    # 1) Full resume (weights + optimizer + scheduler + step + rng)
    resume_ckpt_path = getattr(args, "resume_ckpt_path", None)
    # 2) Weights-only resume (keeps inference compatibility)
    resume_model_path = getattr(args, "resume_model_path", None)

    n_gradient_step = 0
    if resume_ckpt_path is not None and os.path.exists(resume_ckpt_path):
        n_gradient_step = load_full_checkpoint(
            resume_ckpt_path, agent=agent, optimizer=optimizer, scheduler=lr_scheduler, device=args.device
        )
        # Continue from the NEXT step
        n_gradient_step += 1
    elif resume_model_path is not None and os.path.exists(resume_model_path):
        # weights-only: your old format {'model','model_ema'}
        ckpt = torch.load(resume_model_path, map_location="cpu")
        agent.model.load_state_dict(ckpt["model"])
        agent.model_ema.load_state_dict(ckpt["model_ema"])
        print(f"[MODEL] Loaded weights-only: {resume_model_path}")
        n_gradient_step = 0

    if getattr(args, "mode", "train") == "train":
        diffusion_loss_list = []
        start_time = time.time()

        # For resuming: advance dataloader forever; step counter controls saving/eval
        for batch in loop_dataloader(dataloader):
            # get condition
            nobs = batch['obs']
            condition = {k: nobs[k][:, :args.obs_steps, :].to(args.device) for k in nobs.keys()}

            # optional low-dim state noise augmentation (training only)
            if getattr(args, "enable_state_noise", False):
                condition = add_lowdim_obs_noise(
                    condition,
                    shape_meta=args.shape_meta,
                    noise_std=float(getattr(args, "state_noise_std", 0.0)),
                    noise_key_to_std=getattr(args, "state_noise_key_to_std", None),
                )

            naction = batch['action'].to(args.device)

            diffusion_loss = agent.update(naction, condition)['loss']
            lr_scheduler.step()
            diffusion_loss_list.append(diffusion_loss)

            if n_gradient_step % args.log_freq == 0:
                metrics = {
                    'step': n_gradient_step,
                    'total_time': time.time() - start_time,
                    'avg_diffusion_loss': float(np.mean(diffusion_loss_list))
                }
                logger.log(metrics, category='train')
                diffusion_loss_list = []

            # ---- Save both formats ----
            if n_gradient_step % args.save_freq == 0:
                # (A) Old inference format (unchanged)
                model_fp = pathlib.Path(logger._model_dir) / f"model_{str(n_gradient_step)}.pt"
                save_inference_agent_pt(model_fp, agent)

                # (B) New full-resume checkpoint
                ckpt_fp = pathlib.Path(logger._model_dir) / f"ckpt_{str(n_gradient_step)}.pt"
                save_full_checkpoint(
                    ckpt_fp,
                    agent=agent,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    step=n_gradient_step,
                    device=args.device
                )

            if n_gradient_step % args.eval_freq == 0:
                print("Evaluate model...")
                agent.model.eval()
                agent.model_ema.eval()
                # env inference is commented in your original script
                agent.model.train()
                agent.model_ema.train()

            n_gradient_step += 1
            if n_gradient_step > args.gradient_steps:
                # final save
                model_fp = pathlib.Path(logger._model_dir) / "model_final.pt"
                ckpt_fp = pathlib.Path(logger._model_dir) / "ckpt_final.pt"
                save_inference_agent_pt(model_fp, agent)
                save_full_checkpoint(ckpt_fp, agent, optimizer, lr_scheduler, n_gradient_step, args.device)
                logger.finish(agent)
                break

    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    pipeline()