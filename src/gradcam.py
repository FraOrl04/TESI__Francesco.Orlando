# src/gradcam.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from .config import DEVICE

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.fwd_handle = target_layer.register_forward_hook(forward_hook)
        self.bwd_handle = target_layer.register_backward_hook(backward_hook)

    def __del__(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate(self, image_tensor, class_idx=None):
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        self.model.zero_grad()

        output = self.model(image_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        target = output[0, class_idx]
        target.backward()

        # gradients: (B, C, H, W)
        grads = self.gradients  # last conv grads
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP sui gradienti
        cam = (weights * acts).sum(dim=1, keepdim=True)  # somma sui canali

        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam, class_idx


def overlay_gradcam_on_image(image_tensor, cam, out_path):
    """
    image_tensor: (3, H, W), normalizzata [0,1] o simile
    cam: (H, W) numpy
    """
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img -= img.min()
    img /= (img.max() + 1e-8)

    cam_resized = cam
    if cam.shape != img.shape[:2]:
        from cv2 import resize, INTER_LINEAR
        cam_resized = resize(cam, (img.shape[1], img.shape[0]), interpolation=INTER_LINEAR)

    heatmap = plt.cm.jet(cam_resized)[..., :3]
    overlay = 0.4 * heatmap + 0.6 * img
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Immagine")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Grad-CAM")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def generate_gradcam_examples(model, dataloader, target_layer, out_prefix="gradcam", num_examples=8):
    model.eval()
    gradcam = GradCAM(model, target_layer)

    count = 0
    for images, labels in dataloader:
        for i in range(images.size(0)):
            img = images[i]
            cam, cls = gradcam.generate(img)
            overlay_gradcam_on_image(img, cam, f"{out_prefix}_{count}_class{cls}.png")
            count += 1
            if count >= num_examples:
                return
