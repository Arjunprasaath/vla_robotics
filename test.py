# TESTING TEXT AND IMAGE PROCESSING ON VLM

import torch
import numpy as np
import requests
from PIL import Image
import torch.nn as nn
import cv2
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor # combines both tokenizer and image processor

def logits_logprob(logits):
    log_probs = []
    for i in logits:
        prob_i = torch.softmax(i, dim=-1)
        log_prob_i = torch.log(prob_i + 1e-8)
        log_probs.append(log_prob_i)
    return log_probs

model_path = "/projects/p32722/Models/Qwen2-VL-2B-Instruct"
# prompt ="You are in a robotics simulation. The view is from the gripper, looking down at the table. Your task is the pick the object present in the environment up. Analyze the image and explain the scene."
previous_action = None
prompt = f"""
    You are in a robot simulation, your presented with the from a robot gripper.
    Task: Pick and lift the red cube.
    Available actions: forward, backward, up, down, left, right, open gripper, close gripper
    Previous action: {previous_action}
    Do not talk or explain, just output the next 5 action to be taken.
"""
image_path = "/home/sar4384/vla_robotics/example_robot0_robotview.png"
pil_image = Image.open(image_path)
np_image = np.array(pil_image)

processor = Qwen2VLProcessor.from_pretrained(model_path)
model = Qwen2VLForConditionalGeneration.from_pretrained(model_path)
print(f"Loaded model from: {model_path}")
# print(processor.tokenizer.eos_token_id)
messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": prompt}, 
                    {"type": "image", "image": np_image}],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) #return_dict=True, return_tensors="pt")
inputs = processor(text=[text], images=[np_image], return_tensors="pt")
# print(inputs)
# breakpoint()
generation_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=10, output_logits=True, output_scores=True, return_dict_in_generate=True, eos_token_id=processor.tokenizer.eos_token_id)
print(generation_ids.logits[0][-1, :], generation_ids.logits[0][-1, :].shape)
# breakpoint()
logits = generation_ids.logits
log_probs = logits_logprob(logits)
for i in log_probs:
    print(processor.tokenizer.decode(torch.argmax(i)))
# generation_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generation_ids.sequences)
# ]
# print(processor.batch_decode(generation_ids_trimmed, skip_special_token=True, clean_up_tokenization_spaces=False))

generation_sequences = generation_ids.sequences
input_length = inputs.input_ids.shape[1]
generated_token_ids = generation_sequences[0][input_length :]
print(generated_token_ids)
generated_text = processor.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
print(generated_text)
# model.eval()
# model()

# GETTING DIFFERENT TYPES OF CAMERAS AVAILABLE

# import robosuite as suite

# # create environment instance
# env = suite.make(
#     env_name="Lift", # try with other tasks like "Stack" and "Door"
#     robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
#     has_renderer=True,
#     has_offscreen_renderer=False,
#     use_camera_obs=False,
# )

# sim = env.sim
# camera_names = [sim.model.camera_id2name(i) for i in range(sim.model.ncam)]

# print(camera_names)

# # CHECKING FOR VALUE HEAD ON VLA
# import torch
# import torch.nn as nn
# from transformers import Qwen2VLModel, Qwen2VLProcessor

# class VLMWithCustomValueHead(nn.Module):
#     def __init__(self, vlm_model, hidden_size):
#         super().__init__()
#         self.vlm = vlm_model
#         self.value_head = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1)
#         )
    
#     def forward(self, input_ids, attention_mask=None):
#         hidden_output = self.vlm(input_ids, attention_mask=attention_mask)
#         print(hidden_output.last_hidden_state.shape)
#         last_hidden_state = hidden_output.last_hidden_state[:, -1, :]
#         value = self.value_head(last_hidden_state)
#         return hidden_output, value

# model_path =  "/projects/p32722/Models/Qwen2-VL-2B-Instruct"
# model = Qwen2VLModel.from_pretrained(model_path)
# processor = Qwen2VLProcessor.from_pretrained(model_path)
# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": url,
#             },
#             {"type": "text", "text": "Describe the image."},
#         ],
#     }
# ]

# text = processor.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
# )
# print(text)
# inputs = processor(text=[text], images=[url], return_tensors='pt')
# print(inputs)
# # Generate
# # generated_ids = model(**inputs, max_new_tokens=100)
# value_head = VLMWithCustomValueHead(model, model.config.hidden_size)
# hidden_output, value = value_head(inputs.input_ids, inputs.attention_mask)
# print(value)
# print(model.config.output_hidden_states)
# print(model.config.hidden_size)

