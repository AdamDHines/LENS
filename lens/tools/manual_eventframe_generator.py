import torch
import imageio
import os
import numpy as np

def create_images(events, img_folder, img_id):
    if events.any():
        frame = torch.zeros((80, 80), dtype=int)
        for event in events:
            frame[event.y-1, event.x-1] += 1
        imageio.imwrite(f'{img_folder}/frame_{img_id:05d}.png',frame.detach().cpu().numpy().astype(np.uint8))
        print(f'{img_folder}/frame_{img_id:05d}.png')
    else:
        print("No events")

base_folder = '/home/adam/repo/LENS/lens/output'
event_folder = '220724-15-21-59'
files = os.listdir(os.path.join(base_folder, event_folder, 'events'))
files.sort()

output_folder = os.path.join(base_folder, event_folder, 'images')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

events = []
for file in files:
    events.append(np.load(os.path.join(base_folder, event_folder, 'events', file), allow_pickle=True))

for idx, event in enumerate(events):
    create_images(event, output_folder, idx)