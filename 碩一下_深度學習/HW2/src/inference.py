import argparse
import torch
from torch.utils.data import DataLoader

from oxford_pet import *
from utils import *
from models.unet import *
from models.resnet34_unet import * 
from evaluate import * 

import warnings
warnings.filterwarnings("ignore")

def load_model(args, model_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == 'ResNet34':
        model = ResNet34_UNet(in_channels = 3, out_channels = 1)
    else:
        model = UNet(in_channels = 3, out_channels = 1)
        
    model.load_state_dict(torch.load(args.model, map_location = device))
    model = model.to(device)
    return model    


def test(args):
    # args
    data_path = args.data_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    model_path = args.model
    visualize = args.visualize
    
    
    # load data       
    test_data = load_dataset(data_path,mode="test")
    test_data_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)
    
    # load model
    model_type = model_path.split('_')[-3]
    model = load_model(args, model_type)
    
    # evaluate
    dice_score = evaluate(model, test_data_loader, device)
    print(f"Testing dice score:{dice_score}")
    
    # vizualize
    if visualize==1:
        if model_type == 'ResNet34':
            save_dir = f"../result/ResNet"
        else:
            save_dir = f"../result/UNet"
        os.makedirs(save_dir, exist_ok=True)    
    
        model.eval()    
        with torch.no_grad():
            for batch in tqdm(test_data_loader, desc="Generating predictions", ncols=100):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                filenames = batch["filename"] 

                # model predictions
                preds = model(images)
                preds = torch.sigmoid(preds)  
                preds = (preds > 0.5).float()  

                for img, pred, filename in zip(images.cpu().numpy(), preds.cpu().numpy(), filenames):
                    save_path = os.path.join(save_dir, f"{filename}.png")  
                    viz(img, pred, save_path)
    
    

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str,default="../dataset/oxford-iiit-pet", help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--visualize', '-v', type=int, default=0, help='visualize or not')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    test(args)
