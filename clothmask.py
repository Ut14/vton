import os
from PIL import Image
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from networks.u2net import U2NET

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
image_dir = 'inputs/cloth'
result_dir = 'output/masked_cloth'
checkpoint_path = 'checkpoints/cloth_segm_u2net_latest.pth'

# Ensure output directory exists
os.makedirs(result_dir, exist_ok=True)

def load_checkpoint_mgpu(model, checkpoint_path):
    """Loads the model checkpoint for U2-Net."""
    if not os.path.exists(checkpoint_path):
        print("❌ No checkpoint found at the given path!")
        return model

    model_state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = OrderedDict()

    for k, v in model_state_dict.items():
        name = k[7:]  # Remove `module.` if present
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    print("✅ U2-Net Model Loaded Successfully from:", checkpoint_path)
    return model

class NormalizeImage(object):
    """Normalizes image tensors."""
    def __init__(self, mean=0.5, std=0.5):
        self.normalize = transforms.Normalize([mean] * 3, [std] * 3)

    def __call__(self, image_tensor):
        return self.normalize(image_tensor)

# Define image transformation pipeline
transform_rgb = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to 512x512 for better processing
    transforms.ToTensor(),
    NormalizeImage(0.5, 0.5),
])

# Load U2-Net model
net = U2NET(in_ch=3, out_ch=4)
net = load_checkpoint_mgpu(net, checkpoint_path)

def apply_cloth_mask(image_path, output_folder):
    """Applies cloth segmentation mask on a given image and saves it."""
    img = Image.open(image_path).convert('RGB')
    img_size = img.size  
    img = img.resize((512, 512), Image.BICUBIC)

    # Transform image
    image_tensor = transform_rgb(img).unsqueeze(0).to(device)

    # Run the model
    with torch.no_grad():
        output_tensor = net(image_tensor)
    
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.argmax(output_tensor, dim=1, keepdim=True)
    output_tensor = output_tensor.squeeze().cpu().numpy()

    print("🟢 Unique values in output tensor:", np.unique(output_tensor))

    # Convert to binary mask
    output_img = Image.fromarray((output_tensor * 255).astype('uint8'))
    output_img = output_img.resize(img_size, Image.BICUBIC)

    # Ensure valid output file extension
    output_filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
    output_path = os.path.join(output_folder, output_filename)

    # Save the final mask
    output_img.save(output_path)

    print(f"✅ Cloth mask saved: {output_path}")
    
    return output_path  # Ensure this function returns the saved file path

# Process all images in the folder
if __name__ == "__main__":
    for image_name in sorted(os.listdir(image_dir)):
        input_path = os.path.join(image_dir, image_name)
        
        # Skip non-image files
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        output_path = os.path.join(result_dir, os.path.splitext(image_name)[0] + '.png')
        apply_cloth_mask(input_path, output_path)
