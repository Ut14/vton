import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import kornia.geometry.transform as tgm

from datasets import VITONDataset, VITONDataLoader
from network import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, save_images


def test(opt, seg, gmm, alias):
    print("\n🔍 ===== DEBUG MODE ENABLED =====")
    print(f"🛠️ Working directory: {os.getcwd()}")
    print(f"🛠️ CUDA available: {torch.cuda.is_available()}")
    save_path = os.path.join(opt.save_dir, opt.name)
    os.makedirs(save_path, exist_ok=True)
    print(f"🛠️ Output will be saved to: {save_path}")
    print(f"🛠️ Directory exists? {os.path.exists(save_path)}")
    print(f"🛠️ Directory contents: {os.listdir(save_path) if os.path.exists(save_path) else 'N/A'}")
    # Upsampling and blur modules
    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).cuda()

    # Load dataset
    test_dataset = VITONDataset(opt)
    test_loader = VITONDataLoader(opt, test_dataset)

    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            print(f"\n===== Processing Batch {i + 1} =====")

            # Inputs
            img_names = inputs['img_name']
            c_names = inputs['c_name']['unpaired']
            img_agnostic = inputs['img_agnostic'].cuda()
            parse_agnostic = inputs['parse_agnostic'].cuda()
            pose = inputs['pose'].cuda()
            c = inputs['cloth']['unpaired'].cuda()
            cm = inputs['cloth_mask']['unpaired'].cuda()

            # Debug shapes
            print("Input shapes:")
            print(f"  img_agnostic: {img_agnostic.shape}")
            print(f"  parse_agnostic: {parse_agnostic.shape}")
            print(f"  pose: {pose.shape}")
            print(f"  cloth: {c.shape}")
            print(f"  cloth_mask: {cm.shape}")

            # === Fix cloth mask dimensions ===
            if cm.dim() == 2:
                cm = cm.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            elif cm.dim() == 3:
                cm = cm.unsqueeze(0)  # [1, C, H, W]
            elif cm.dim() == 4:
                pass  # already correct
            else:
                raise ValueError(f"Unexpected cm dimensions: {cm.shape}")
                
            if cm.shape[1] == 1:
                cm = cm.repeat(1, 3, 1, 1)


            # === Part 1: Segmentation Generation ===
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
            c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
            noise = gen_noise(cm_down.size()).cuda()

            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, noise), dim=1)
            print("Segmentation network input shape:", seg_input.shape)

            parse_pred_down = seg(seg_input)
            parse_pred = gauss(up(parse_pred_down))
            parse_pred = parse_pred.argmax(dim=1)[:, None]
            print("Segmentation prediction shape:", parse_pred.shape)

            # One-hot encode segmentation prediction
            parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float).cuda()
            parse_old.scatter_(1, parse_pred, 1.0)

            # Group segmentation labels
            labels = {
                0: ['background', [0]],
                1: ['paste', [2, 4, 7, 8, 9, 10, 11]],
                2: ['upper', [3]],
                3: ['hair', [1]],
                4: ['left_arm', [5]],
                5: ['right_arm', [6]],
                6: ['noise', [12]],
            }

            parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float).cuda()
            for j in range(len(labels)):
                for label in labels[j][1]:
                    parse[:, j] += parse_old[:, label]

            print("Parse map (7 parts) shape:", parse.shape)

            # === Part 2: Clothes Deformation (GMM) ===
            agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
            parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
            pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
            c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')

            gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)
            print("GMM input shape:", gmm_input.shape)

            _, warped_grid = gmm(gmm_input, c_gmm)
            warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

            print("Warped cloth shape:", warped_c.shape)
            print("Warped cloth mask shape:", warped_cm.shape)

            # === Part 3: Try-On Synthesis (ALIAS) ===
            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask = torch.clamp(misalign_mask, min=0.0)
            parse_div = torch.cat((parse, misalign_mask), dim=1)
            parse_div[:, 2:3] -= misalign_mask

            alias_input = torch.cat((img_agnostic, pose, warped_c), dim=1)
            output = alias(alias_input, parse, parse_div, misalign_mask)
            print("Final synthesized image shape:", output.shape)

            print("\n💾 Attempting to save results...")
    unpaired_names = []
    for img_name, c_name in zip(img_names, c_names):
        # Ensure proper filename extension
        unpaired_name = f"{os.path.splitext(img_name)[0]}_{os.path.splitext(c_name)[0]}.jpg"
        unpaired_names.append(unpaired_name)
        print(f"  Will save: {unpaired_name}")

    # Verify output tensors before saving
    print("\n🔍 Output tensor inspection:")
    print(f"Shape: {output.shape}")
    print(f"Min: {output.min().item():.4f}, Max: {output.max().item():.4f}")
    print(f"Mean: {output.mean().item():.4f}, STD: {output.std().item():.4f}")

    # Save images
    try:
        save_images(output, unpaired_names, save_path)
        print(f"✅ Successfully saved {len(unpaired_names)} images")
        
        # Verify files were actually written
        saved_files = os.listdir(save_path)
        print(f"🛠️ Directory now contains {len(saved_files)} files")
        for f in saved_files[-3:]:  # Show last 3 files
            print(f"  - {f} ({os.path.getsize(os.path.join(save_path, f))} bytes)")
    except Exception as e:
        print(f"❌ Failed to save images: {str(e)}")
        raise
    # ===== VISUAL DEBUGGING =====
    try:
        import matplotlib.pyplot as plt
        debug_path = "debug_visualization.png"
        
        plt.figure(figsize=(18,6))
        
        # Input Cloth
        plt.subplot(1,4,1)
        plt.imshow(c[0].permute(1,2,0).cpu().numpy()*0.5+0.5)
        plt.title('Input Cloth')
        
        # Warped Cloth
        plt.subplot(1,4,2)
        plt.imshow(warped_c[0].permute(1,2,0).cpu().numpy()*0.5+0.5)
        plt.title('Warped Cloth')
        
        # Model Input
        plt.subplot(1,4,3)
        plt.imshow(img_agnostic[0].permute(1,2,0).cpu().numpy()*0.5+0.5)
        plt.title('Model Input')
        
        # Final Output
        plt.subplot(1,4,4)
        plt.imshow(output[0].permute(1,2,0).cpu().numpy()*0.5+0.5)
        plt.title('Final Output')
        
        plt.tight_layout()
        plt.savefig(debug_path)
        print(f"✅ Debug visualization saved to {debug_path}")
    except Exception as e:
        print(f"⚠️ Could not create debug visualization: {str(e)}")
