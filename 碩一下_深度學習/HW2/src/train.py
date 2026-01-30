import argparse
from oxford_pet import *
from utils import * 
from models.unet import *
from models.resnet34_unet import *
from evaluate import * 
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam
import random


def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  

def train(args):
    # implement the training function here
    # args
    data_path = args.data_path
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    model_type = args.model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    train_writer = SummaryWriter(f"log/{model_type}/train_lr{lr}")    
    valid_writer = SummaryWriter(f"log/{model_type}/valid_lr{lr}")
    if model_type == 'unet':
        model_name = 'UNet'
    else:
        model_name = 'ResNet34_UNet'
    
    # data
    train_data = load_dataset(data_path,mode="train")
    train_data_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    
    valid_data = load_dataset(data_path,mode="valid")
    valid_data_loader = DataLoader(valid_data, batch_size = batch_size, shuffle = False)
    
    # model
    if model_type == 'unet':
        model = UNet(in_channels = 3, out_channels = 1).to(device)
    else:
        model = ResNet34_UNet(in_channels = 3, out_channels = 1).to(device)
    
    # loss
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr = lr)
    
    # training
    for epoch in range(epochs):
        running_loss = 0.0
        for _, sample in enumerate(tqdm(train_data_loader, desc="Training Progress")):
            image = sample['image'].to(device)
            mask = sample['mask'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(image)
            outputs = outputs.flatten(start_dim=1, end_dim = 3)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()
            
            running_loss +=loss.item()
        
        train_writer.add_scalar('training loss', running_loss/ len(train_data_loader), epoch)
        print(f"Epoch:{epoch+1}, Loss:{running_loss/ len(train_data_loader)}")
        
        train_score = evaluate(model, train_data_loader, device)
        train_writer.add_scalar('dice score', train_score, epoch)
        print(f"Training dice score:{train_score}")        
        valid_score = evaluate(model, valid_data_loader, device)
        valid_writer.add_scalar('dice score', valid_score, epoch)
        print(f"Validation dice score:{valid_score}")
        
        if (epoch >= epochs - 5):
            torch.save(model.state_dict(), f"../saved_models/DL_Lab2_{model_name}_{epoch}.pth")
        

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str,default="../dataset/oxford-iiit-pet", help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=150, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--model', '-m', type=str, default="unet", choices=['unet', 'resnet34_unet'])

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    set_seed(42)
    train(args)
    
    
    