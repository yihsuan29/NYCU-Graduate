import matplotlib.pyplot as plt
import numpy as np

def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    pred_mask = pred_mask>0.5
    pred_mask = pred_mask.cpu().numpy().astype(np.float32)
    gt_mask = gt_mask.cpu().numpy().astype(np.float32)
    
    pred = pred_mask.flatten()
    gt = gt_mask
    
    common_pixel = np.sum(pred*gt)
    pred_size = np.sum(pred)
    gt_size = np.sum(gt)
    
    dice_score  = 2* common_pixel/(pred_size + gt_size)
    
    return dice_score 


def viz(image, mask, save_path=None):  
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    image = image * 255.0
    image = np.moveaxis(image, 0, -1).astype(np.uint8)
    
    axes[0].imshow(image)
    axes[0].set_title("Image")
    
    image_height, image_width = image.shape[:2]
    mask = mask.reshape((image_height, image_width))        
    mask = mask.squeeze()
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask")     
    
    overlay = np.copy(image)
    overlay[mask == 1] = [255, 255, 255] 
    
    axes[2].imshow(image)
    axes[2].imshow(overlay, alpha=0.5)
    axes[2].set_title("Image with Mask Overlay")  
    
    if save_path:
        plt.axis("off")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

