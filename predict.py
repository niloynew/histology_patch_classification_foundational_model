import argparse
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image

def load_image(path, image_size=224):
    tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    return tf(Image.open(path).convert("RGB")).unsqueeze(0)

def main(ckpt_path, input_path, device="cpu"):
    # Load saved task (contains encoder+squeezer+classifier)
    exported_task = torch.load(ckpt_path, map_location=device)
    exported_task.eval()

    # Class names
    class_names = exported_task.task.class_names
    print("Loaded classes:", class_names)

    # Collect files
    p = Path(input_path)
    files = [p] if p.is_file() else [f for f in p.glob("**/*") if f.suffix.lower() in {".tif",".tiff"}]

    for f in files:
        x = load_image(f).to(device)
        with torch.inference_mode():
            logits = exported_task.forward((x, None))  # (image, mil_indices=None)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(probs.argmax())
            print(f"{f.name}: {class_names[pred_idx]} (prob={probs[pred_idx]:.3f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="trainer_checkpoints/pretrained_exp/bestbyvalidation-1", help="Path to trained task checkpoint")
    ap.add_argument("input", help="Image file or folder")
    args = ap.parse_args()
    main(args.ckpt, args.input)
