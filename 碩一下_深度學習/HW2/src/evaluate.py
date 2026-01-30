import torch
from utils import *
def evaluate(net, data, device):
    # implement the evaluation function here
    score = 0
    image_num = 0
    with torch.no_grad():
        for _, sample in enumerate(data):
            image = sample['image'].to(device)
            mask = sample['mask'].to(device)
            outputs = net(image)
            output_num = outputs.shape[0]
            for i in range(output_num):
                score += dice_score(outputs[i], mask[i])
            image_num += output_num
    
    return score/image_num