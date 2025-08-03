import torch
import torch.nn as nn
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# The UNET and DoubleConv class definitions should be here, same as before
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNET, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(256, 512)
        self.down4 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.conv1(x2)
        x3 = self.down2(x2)
        x3 = self.conv2(x3)
        x4 = self.down3(x3)
        x4 = self.conv3(x4)
        x5 = self.down4(x4)
        x5 = self.conv4(x5)
        u1 = self.up1(x5)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.upconv1(u1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.upconv2(u2)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.upconv3(u3)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.upconv4(u4)
        logits = self.outc(u4)
        return logits


# Make sure this function is present
def process_image(model, image_bgr, saturation_reduction, brightness_increase):
    # Get mask
    original_h, original_w, _ = image_bgr.shape
    transform = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image_rgb)
    input_tensor = transformed["image"].unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy().squeeze()
    
    # Resize probability map and create binary mask
    probability_map_resized = cv2.resize(probabilities, (original_w, original_h))
    binary_mask = (probability_map_resized > 0.5).astype(np.uint8)

    # Create visualization images
    heatmap_gray = (probability_map_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_VIRIDIS)
    blended_image = cv2.addWeighted(heatmap_color, 0.5, image_bgr, 0.5, 0)
    
    # Apply whitening filter
    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s = s.astype(float)
    v = v.astype(float)
    s = np.where(binary_mask == 1, s - saturation_reduction, s)
    v = np.where(binary_mask == 1, v + brightness_increase, v)
    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge([h, s.astype(np.uint8), v.astype(np.uint8)])
    whitened_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    return binary_mask * 255, heatmap_color, blended_image, whitened_image