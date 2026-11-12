import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[:, 0]
        target.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback uniform heatmap if hooks didn't trigger
            return np.zeros((14, 14), dtype=np.float32)

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]

        # Handle CNNs vs Vision Transformers
        if len(gradients.shape) == 3:
            weights = np.mean(gradients, axis=(1, 2))
            cam = np.zeros(activations.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * activations[i]
        elif len(gradients.shape) == 2:
            weights = np.mean(gradients, axis=0)
            cam = np.dot(activations, weights)
            if cam.shape[0] == 197:
                cam = cam[1:].reshape(14, 14)
            else:
                side = int(np.sqrt(cam.shape[0]))
                cam = cam.reshape(side, side) if side * side == cam.shape[0] else cam.reshape(14, 14)
        else:
            cam = np.mean(gradients, axis=0)

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        else:
            cam = np.zeros((14, 14), dtype=np.float32)
            
        return np.array(cam, dtype=np.float32)

def overlay_heatmap(image_np, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    if image_np.dtype != np.uint8:
        image_np = np.uint8(image_np)
    
    # Ensure heatmap is a valid 2D numpy array
    if not isinstance(heatmap, np.ndarray) or heatmap.ndim != 2:
        heatmap = np.zeros(image_np.shape[:2], dtype=np.float32)

    h, w = image_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
    superimposed = cv2.addWeighted(image_np, 1 - alpha, colored_heatmap, alpha, 0)
    return superimposed