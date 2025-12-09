# VLA Robotics

Vision-Language-Action (VLA) robotics training using PPO reinforcement learning with Qwen2-VL-2B-Instruct.

## Overview

Trains a vision-language model to control a robot in simulation using Proximal Policy Optimization (PPO). The model generates text-based actions from visual observations to complete the Lift task in Robosuite.

## Features

- **Vision-Language Model**: Qwen2-VL-2B-Instruct for action generation from images
- **PPO Training**: Actor-critic with Generalized Advantage Estimation (GAE)
- **Robosuite Integration**: Panda robot in Lift environment
- **Wandb Logging**: Experiment tracking and metrics visualization

## Task

Pick and lift the red cube using discrete actions: forward, backward, up, down, left, right, open gripper, close gripper.

## Usage

### Training

python main.py

#### Configuration

Key parameters in `main.py`:
- `epochs`: Number of training epochs (default: 10)
- `episodes`: Episodes per epoch (default: 5)
- `total_steps`: Max steps per episode (default: 100)
- `LR`: Learning rate (default: 5e-5)
- `mini_batch_size`: Mini-batch size for PPO updates (default: 25)

### Inference

Uncomment the `vla_inference()` call to run inference with a trained model.

## Requirements

See `pyproject.toml` for dependencies. Key packages:
- PyTorch
- Transformers (Qwen2-VL)
- Robosuite
- Wandb

## Model Paths

Update `model_path` and `save_path` in `main.py` to point to your model directories.