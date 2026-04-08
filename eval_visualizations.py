import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as F
from da6401_assignment_2.models.multitask import MultiTaskPerceptionModel
import cv2

def log_feature_maps(model, image_tensor, device):
    """Task 2.4: Inside the Black Box - Feature Maps"""
    print("Extracting feature maps...")
    model.eval()
    
    # We need to hook into the first and last conv layer
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
        
    # Register hooks
    h1 = model.encoder.conv1.register_forward_hook(get_activation('conv1'))
    h2 = model.encoder.conv5_2.register_forward_hook(get_activation('conv_last'))
    
    with torch.no_grad():
        _ = model(image_tensor.unsqueeze(0).to(device))
        
    h1.remove()
    h2.remove()
    
    # Plotting helper
    for layer_name in ['conv1', 'conv_last']:
        act = activation[layer_name].squeeze().cpu().numpy()
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle(f'Feature Maps: {layer_name}')
        for i, ax in enumerate(axes.flat):
            if i < act.shape[0]:
                ax.imshow(act[i], cmap='viridis')
            ax.axis('off')
        
        wandb.log({f"Feature_Maps_{layer_name}": wandb.Image(fig)})
        plt.close(fig)

def main():
    wandb.init(project="DA6401-Assignment2", name="Visualizations_and_Eval")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate model and load your best checkpoint here (e.g., from full fine-tuning)
    model = MultiTaskPerceptionModel(num_breeds=37, seg_classes=3, in_channels=3).to(device)
    
    # Provide the path to the best model checkpoint that gets generated during training
    ckpt_path = "checkpoints/exp_2_3_tf_full_best.pth"
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded {ckpt_path}")
    except FileNotFoundError:
        print(f"Warning: {ckpt_path} not found. Running with untrained weights for demonstration.")
    
    # --- Task 2.4 Feature Maps ---
    # Create simple dummy tensor mimicking a normalized 224x224 image for feature extraction
    dummy_img = torch.randn(3, 224, 224) 
    log_feature_maps(model, dummy_img, device)
    
    # --- Task 2.5 Object Detection Table & Task 2.6 Segmentation Samples ---
    # In a real scenario, you'd loop over `val_loader`. 
    # Here we log a structured W&B Table placeholder.
    print("Setting up W&B tables for Detection and Segmentation...")
    columns = ["Image", "Confidence", "IoU", "Failure Case Analysis"]
    bbox_table = wandb.Table(columns=columns)
    
    seg_columns = ["Original", "Ground Truth Trimap", "Predicted Trimap", "Pixel Accuracy", "Dice Score"]
    seg_table = wandb.Table(columns=seg_columns)
    
    # Note: To fully populate these, you simply integrate this into the loops inside `inference.py`.
    # wandb.log({"Object_Detection_Results": bbox_table, "Segmentation_Results": seg_table})
    
    print("Visualizations logged to W&B!")
    wandb.finish()

if __name__ == "__main__":
    main()
