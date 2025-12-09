import os
import cv2
import wandb
import torch
import imageio
import numpy as np
from PIL import Image
import torch.nn as nn
import robosuite as suite
import torch.optim as optim
import torch.nn.functional as F
import robosuite.macros as macros
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, Qwen2VLProcessor, Qwen2VLModel

class VLMWithCustomValueHead(nn.Module):
    def __init__(self, vlm_model, hidden_size):
        super().__init__()
        self.vlm = vlm_model
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        vlm_device = next(vlm_model.parameters()).device
        vlm_dtype = next(vlm_model.parameters()).dtype
        self.value_head = self.value_head.to(device=vlm_device, dtype=vlm_dtype)
    
    def forward(self, input_ids, attention_mask=None):
        hidden_output = self.vlm(input_ids, attention_mask=attention_mask)
        last_hidden_state = hidden_output.last_hidden_state[:, -1, :]
        value = self.value_head(last_hidden_state)
        return hidden_output, value

def generalized_advantage_estimate(rewards, dones, state_values):
    lambda_gae = 0.95
    gamma = 0.99
    last_gae = 0.0
    gae = torch.zeros_like(rewards) # [total_steps]
    returns = torch.zeros_like(rewards) # [total_steps]
    for i in reversed(range(len(rewards))):
        # mask is zero when done, else 1
        mask = 1.0 - dones[i]
        # calculate TD error: delta[i] = reward[i] + gamma * Vs(i + 1) * mask - Vs(i)
        delta = rewards[i] + gamma * state_values[i + 1] * mask - state_values[i]
        # calculate advantage: A_i = delta_i + gamma * lambda * A(i + 1) *  mask
        gae[i] = delta + gamma * lambda_gae * last_gae * mask
        last_gae = gae[i]
        # calculate return = Advantage + state value
        returns[i] = gae[i] + state_values[i]

    # mean_gae = torch.mean(gae, dim=0)
    # std_gae = torch.std(gae, dim=0) + 1e-8
    # normalized_gae = (gae - mean_gae) / std_gae
    # print(gae.shape, returns.shape, rewards, gae, returns)
    return gae, returns

def ppo(actor_model, value_model, episode_buffer, prompt, actor_optimizer, value_optimizer, processor, mini_batch_size=10):
    epsilon = 0.2
    entropy_coef = 0.10
    device = next(actor_model.parameters()).device

    actor_model.train()
    value_model.train()

    num_steps = len(episode_buffer['rewards'])
    total_actor_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    num_updates = 0

    advantages = episode_buffer['gae']
    normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor_optimizer.zero_grad()
    value_optimizer.zero_grad()

    for i in range(num_steps): # for each step
        # Input for actor
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": episode_buffer['obs'][i]}]}, 
            {"role": "assistant", "content": [{"type": "text", "text": episode_buffer['action_texts'][i]}]}
            ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[episode_buffer['obs'][i]], return_tensors="pt").to(device)
        
        # Actor model forward pass
        outputs = actor_model(**inputs)
        logits = outputs.logits
        
        # Prompt length
        pormpt_only_messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": episode_buffer['obs'][i]}]}]
        prompt_only_text = processor.apply_chat_template(pormpt_only_messages, tokenize=False, add_generation_prompt=True)
        prompt_only_inputs = processor(text=[prompt_only_text], images=[episode_buffer['obs'][i]], return_tensors="pt").to(device)
        prompt_length = prompt_only_inputs.input_ids.shape[1]

        # Value model forward pass
        _, state_value = value_model(prompt_only_inputs.input_ids, prompt_only_inputs.attention_mask)

        # New log probs
        action_logits = logits[:, prompt_length - 1 : -1, :]
        action_ids = inputs.input_ids[:, prompt_length :]
        log_probs = F.log_softmax(action_logits, dim = -1)
        selected_log_probs = log_probs.gather(dim=-1, index=action_ids.unsqueeze(-1)).squeeze(-1)
        total_log_probs = selected_log_probs.sum(dim=-1)

        # Entropy
        probs = F.softmax(action_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # PPO = -mean(min (ratio * advantage, clip(ratio, 1 - epsilon, 1 + epsilon) * advantage))
        advantage = normalized_advantages[i].to(device)
        old_log_prob = episode_buffer['log_probs'][i].to(device)

        ratio = torch.exp(total_log_probs - old_log_prob)
        ratio = torch.clamp(ratio, 0.0, 10.0)

        term_1 = ratio * advantage
        term_2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
        actor_loss = - torch.min(term_1, term_2).mean()
        
        # Entropy bonus
        step_actor_loss = actor_loss - entropy_coef * entropy

        # Value loss = MSE(state value, returns)
        returns = episode_buffer['returns'][i].to(device)
        returns = returns.to(dtype=state_value.dtype)
        step_value_loss = F.mse_loss(state_value.squeeze(), returns)

        # Scale losses by mini_batch_size for proper gradient accumulation
        scaled_actor_loss = step_actor_loss / mini_batch_size
        scaled_value_loss = step_value_loss / mini_batch_size

        # backward pass
        scaled_actor_loss.backward()
        scaled_value_loss.backward()

        # Accumulate
        total_actor_loss += step_actor_loss.item()
        total_value_loss += step_value_loss.item()
        total_entropy += entropy.item()

        del inputs, outputs, logits, prompt_only_inputs, action_logits, selected_log_probs
        del advantage, old_log_prob, ratio, term_1, term_2, returns, state_value
        del step_actor_loss, step_value_loss, scaled_actor_loss, scaled_value_loss
        del probs, entropy, actor_loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if (i + 1) % mini_batch_size == 0 or (i + 1) == num_steps:
            # Clip gradients
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor_model.parameters(), max_norm=0.7)
            value_grad_norm = torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=0.7)

            # Update weights
            actor_optimizer.step()
            value_optimizer.step()

            # Clear gradients
            actor_optimizer.zero_grad()
            value_optimizer.zero_grad()

            num_updates += 1
            wandb.log({
                "train/actor_grad_norm": actor_grad_norm.item(),
                "train/value_grad_norm": value_grad_norm.item(),
                "train/update_step": num_updates,
            })

            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Average metrics over all steps
    avg_actor_loss = total_actor_loss / num_steps
    avg_value_loss = total_value_loss / num_steps
    avg_entropy = total_entropy / num_steps

    # Log losses and diagnostics
    log_dict = {
        "train/actor_loss": avg_actor_loss,
        "train/value_loss": avg_value_loss,
        "train/entropy": avg_entropy,
        "train/num_weight_updates": num_updates,
    }
    wandb.log(log_dict)

    print(f"Episode - Actor loss: {avg_actor_loss:.4f}, Value loss: {avg_value_loss:.4f}, Entropy: {avg_entropy:.4f}")
    return avg_actor_loss, avg_value_loss

def logits_to_logprob(logits, generated_sequences, input_length, dont_detach = False):
    log_probs, log_probs_ids = [], []
    new_token_ids = generated_sequences[:, input_length :] # [batch, total_len] -> [batch, new_tokens]

    for i, logit_step in enumerate(logits):
        log_prob_i = F.log_softmax(logit_step, dim=-1) # [batch, log_probs]
        chosen_ids = new_token_ids[:, i].unsqueeze(-1) # [chosen_id] -> [1, chosen_id]
        chosen_log_probs = log_prob_i.gather(dim=-1, index=chosen_ids).squeeze(-1) # [1, log_prob] -> [log_prob]
        log_probs.append(chosen_log_probs)

    total_log_probs = torch.stack(log_probs).sum()
    return torch.stack(log_probs), total_log_probs

def load_model(model_path):
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype="auto", device_map="auto")
    headless_model = Qwen2VLModel.from_pretrained(model_path, dtype="auto", device_map="auto")
    processor = Qwen2VLProcessor.from_pretrained(model_path)
    return model, headless_model, processor

def vlm_action(model, processor, input_image, previous_action=None):
    prompt = f"""
        You are in a robot simulation, your presented with the from a robot gripper.
        Task: Pick and lift the red cube.
        Available actions: forward, backward, up, down, left, right, open gripper, close gripper
        Previous action: {previous_action}
        Do not talk or explain, just output the action to be taken.
    """
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}, 
                        {"type": "image", "image": input_image}],
        }
    ]
    # print(prompt)
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    generation_ids = model.generate(**inputs, max_new_tokens=256)
    generation_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generation_ids)
    ]
    output_text = processor.batch_decode(generation_ids_trimmed, skip_special_token=True, clean_up_tokenization_space=False)
    return output_text

def vla_training(env, model_path, video_path, total_steps, skip_steps, action_map):
    pass

def vla_inference(env, model_path, video_path, total_steps, skip_steps, action_map):
    model, _, processor = load_model(model_path)
    obs = env.reset()
    writer = imageio.get_writer(video_path, fps=30)
    vla_choice = None

    for i in range(total_steps):
        vla_choice = vlm_action(model, processor, obs[camera_name + "_image"], vla_choice)
        # print(f"VLA model output: {vla_choice}")
        if "forward" in vla_choice[0]:
            action = action_map["forward"]
        elif "backward" in vla_choice[0]:
            action = action_map["backward"]
        elif "up" in vla_choice[0]:
            action = action_map["up"]
        elif "down" in vla_choice[0]:
            action = action_map["down"]
        elif "left" in vla_choice[0]:
            action = action_map["left"]
        elif "right" in vla_choice[0]:
            action = action_map["right"]
        elif "open gripper" in vla_choice[0]:
            action = action_map["open gripper"]
        elif "close gripper" in vla_choice[0]:
            action = action_map["close gripper"]
        else:
            print("VLA model didn't choose an action")
            print(f"VLA output: {vla_choice}")
            action = action_map['forward']

        if i % skip_steps == 0: 
            for _ in range(skip_steps):
                obs, reward, done, info = env.step(action)
                frame = obs[camera_name + "_image"]
                writer.append_data(frame)
                # print(f"Action:{action}, Reward: {reward}")            
                if done:
                    break
        if done:
            break

    writer.close()
    print(f"Video saved to {video_path}")
    return None

macros.IMAGE_CONVENTION = "opencv"
camera_name = "robot0_robotview" # "robot0_eye_in_hand"
video_training_path = "./video_training.mp4"
video_inference_path = "./video_inference.mp4"
model_path = "/projects/p32722/Models/Qwen2-VL-2B-Instruct"
save_path = "/projects/p32722/Models/vla_robotics_model/"

total_steps = 100
skip_steps = 5
episodes = 5
epochs = 10
LR = 5e-5
mini_batch_size = 25

action_map = {
    'backward': np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),   # Negative x (up key) FORWARD
    'forward': np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),   # Positive x (down key) BACKWARD
    'right': np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),      # Negative y (left key) LEFT
    'left': np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),   # Positive y (right key) RIGHT
    'up': np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),   # UP
    'down': np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]),   # DOWN
    'close gripper': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),   # CLOSE GRIPPER
    'open gripper': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),   # OPEN GRIPPER
}

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    use_camera_obs=True,
    use_object_obs=False,
    has_offscreen_renderer=True,
    camera_names=camera_name,
    camera_heights=256,
    camera_widths=256,
    reward_shaping=True,
)

# vla_inference(env, model_path, video_inference_path, total_steps, skip_steps, action_map)
model, headless_model, processor = load_model(model_path)
value_model = VLMWithCustomValueHead(headless_model, headless_model.config.hidden_size)

# Optimizer
model_optimizer = optim.AdamW(model.parameters(), lr=LR)
value_model_optimizer = optim.AdamW(value_model.parameters(), lr=LR)

previous_action = None
prompt = f"""
    You are in a robot simulation, your presented with the from a robot gripper.
    Task: Pick and lift the red cube.
    Available actions: forward, backward, up, down, left, right, open gripper, close gripper
    Previous action: {previous_action}
    Do not talk or explain, just output the action to be taken.
"""

print("-" * 60)
print("Training Configuration:")
print(f"  Epochs: {epochs}")
print(f"  Episodes per epoch: {episodes}")
print(f"  Max steps per episode: {total_steps}")
print(f"  Mini-batch size: {mini_batch_size}")
print(f"  Weight updates per episode: {total_steps // mini_batch_size}")
print(f"  Total max steps (epochs x episodes x max_steps): {epochs * episodes * (total_steps // mini_batch_size)}")
print("=" * 60)

wandb.init(
    project="vla-robotics",
    name=f"ppo-training-{epochs}epochs-{episodes}episodes-mb{mini_batch_size}",
    config={
        "epochs": epochs,
        "episodes_per_epoch": episodes,
        "total_steps": total_steps,
        "skip_steps": skip_steps,
        "learning_rate": LR,
        "mini_batch_size": mini_batch_size,
        "entropy_coef": 0.10,
        "model_path": model_path,
        "camera_name": camera_name,
        "env_name": "Lift",
        "robot": "Panda",
    }
)

loss = {"actor": [], "value": []}

for epoch in range(epochs):
    epoch_actor_losses = []
    epoch_value_losses = []
    episode_rewards = []
    episode_lengths = []
    epoch_action_names = []

    for episode in range(episodes):
        obs = env.reset()
        episode_buffer = {'obs': [], 'action_texts': [], 'rewards': [], 'state_values': [], 'dones': [], 'log_probs': []}
        if episode == 0 and epoch == 0:
            cv2.imwrite(f"examples/example_{camera_name}.png", obs[camera_name + "_image"])

        episode_reward = 0.0
        episode_length = 0

        for step in range(total_steps):
            with torch.no_grad():
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": obs[camera_name + "_image"]}],}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[obs[camera_name + "_image"]], return_tensors="pt").to(model.device)
            
                # generate action log probs with model
                generation_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=10, output_logits=True, output_scores=True, return_dict_in_generate=True, eos_token_id=processor.tokenizer.eos_token_id)
                generation_sequences = generation_ids.sequences
                input_length = inputs.input_ids.shape[1]
                log_probs, total_log_probs = logits_to_logprob(generation_ids.logits, generation_sequences, input_length)
                
                # generate state value with value model
                _, state_value = value_model(inputs.input_ids, inputs.attention_mask)

                # decode the action
                generated_token_ids = generation_sequences[0, input_length :]
                generated_text = processor.tokenizer.decode(generated_token_ids,)# skip_special_tokens=True)
            
            if "forward" in generated_text:
                action = action_map["forward"]
                action_name = "forward"
            elif "backward" in generated_text:
                action = action_map["backward"]
                action_name = "backward"
            elif "up" in generated_text:
                action = action_map["up"]
                action_name = "up"
            elif "down" in generated_text:
                action = action_map["down"]
                action_name = "down"
            elif "left" in generated_text:
                action = action_map["left"]
                action_name = "left"
            elif "right" in generated_text:
                action = action_map["right"]
                action_name = "right"
            elif "open gripper" in generated_text:
                action = action_map["open gripper"]
                action_name = "open gripper"
            elif "close gripper" in generated_text:
                action = action_map["close gripper"]
                action_name = "close gripper"

            epoch_action_names.append(action_name)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            episode_buffer['obs'].append(obs[camera_name + "_image"].copy())
            episode_buffer['action_texts'].append(generated_text)
            episode_buffer['rewards'].append(reward)
            episode_buffer['dones'].append(float(done))
            episode_buffer['state_values'].append(state_value.detach().cpu().squeeze())
            episode_buffer['log_probs'].append(total_log_probs.detach().cpu())

            del generation_ids, generation_sequences, log_probs, total_log_probs, state_value, inputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            if done or step == total_steps - 1:
                # setting done to True
                episode_buffer['dones'][-1] = float(True)
                with torch.no_grad():
                    # try to use a function for this so that the code is not repeated
                    # using the last observation to get total_step + 1 state values.
                    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": obs[camera_name + "_image"]}],}]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[obs[camera_name + "_image"]], return_tensors="pt").to(model.device)
                    _, state_value = value_model(inputs.input_ids, inputs.attention_mask)
                    episode_buffer['state_values'].append(state_value.detach().cpu().squeeze())
                    del inputs, state_value
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode completed - Reward: {episode_reward:.2f}, Length: {episode_length}")
        
        rewards = torch.tensor(episode_buffer['rewards'])
        dones = torch.tensor(episode_buffer['dones'])
        state_values = torch.stack(episode_buffer['state_values'])

        gae, returns = generalized_advantage_estimate(rewards, dones, state_values)
        episode_buffer['gae'] = gae
        episode_buffer['returns'] = returns
        episode_buffer['log_probs'] = torch.stack(episode_buffer['log_probs'])

        print("Starting PPO update...")
        model_loss, value_loss = ppo(model, value_model, episode_buffer, prompt, model_optimizer, value_model_optimizer, processor, mini_batch_size)

        epoch_actor_losses.append(model_loss)
        epoch_value_losses.append(value_loss)

        wandb.log({
            "episode/reward": episode_reward,
            "episode/length": episode_length,
            "episode/actor_loss": model_loss,
            "episode/value_loss": value_loss,
            "episode/episode_num": episode,
            "epoch": epoch,
        })

        del episode_buffer, rewards, dones, state_values, gae, returns
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    unique_actions = sorted(set(epoch_action_names))

    avg_episode_reward = np.mean(episode_rewards)
    avg_episode_length = np.mean(episode_lengths)
    avg_actor_loss = np.mean(epoch_actor_losses)
    avg_value_loss = np.mean(epoch_value_losses)

    loss['actor'].append(avg_actor_loss)
    loss['value'].append(avg_value_loss)

    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1} Summary:")
    print(f"  Average Reward: {avg_episode_reward:.2f}")
    print(f"  Average Length: {avg_episode_length:.1f}")
    print(f"  Average Actor Loss: {avg_actor_loss:.4f}")
    print(f"  Average Value Loss: {avg_value_loss:.4f}")
    print(f"  Unique Actions: {len(unique_actions)}: {unique_actions}")
    print(f"{'='*60}\n")

    wandb.log({
        "epoch/avg_episode_reward": avg_episode_reward,
        "epoch/avg_episode_length": avg_episode_length,
        "epoch/avg_actor_loss": avg_actor_loss,
        "epoch/avg_value_loss": avg_value_loss,
        "epoch/total_episodes": len(episode_rewards),
        "epoch/unique_actions_count": len(unique_actions),
        "epoch": epoch,
    })

# Save the final trained actor model
print("\n" + "="*60)
print("Saving trained model...")
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
processor.save_pretrained(save_path)
print(f"Model saved to {save_path}")
print("="*60)

wandb.finish()
print("\n" + "="*60)
print("Training Complete!")
print("="*60)