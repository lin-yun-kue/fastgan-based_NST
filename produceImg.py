import torch
import os
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_path = os.path.abspath("models/all_50000.pth")  # 確保這個檔案存在
model_path = os.path.abspath("fine_tuned_stylegan2.pth")  # 確保這個檔案存在
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型權重檔案不存在: {model_path}")


# latent_dim = 100  # 你的 Generator 使用的是 nz=100
netG = Generator(im_size=256, nz=256)
netG.apply(weights_init)

netD = Discriminator(im_size=256)
netD.apply(weights_init)

netG.to(device)
netD.to(device)

ckpt = torch.load(model_path, map_location=device, weights_only=True)
netG.load_state_dict(ckpt['g'])
netD.load_state_dict(ckpt['d'])
# avg_param_G = ckpt['g_ema']
# load_params(netG, avg_param_G)


num_samples = 4  # 產生 4 張圖片
z = torch.randn(num_samples, 256).to(device)  

# def get_early_features(net, noise):
#     feat_4 = net.init(noise)
#     feat_8 = net.feat_8(feat_4)
#     feat_16 = net.feat_16(feat_8)
#     feat_32 = net.feat_32(feat_16)
#     feat_64 = net.feat_64(feat_32)
#     return feat_8, feat_16, feat_32, feat_64

# def get_late_features(net, im_size, feat_64, feat_8, feat_16, feat_32):
#     feat_128 = net.feat_128(feat_64)
#     feat_128 = net.se_128(feat_8, feat_128)

#     feat_256 = net.feat_256(feat_128)
#     feat_256 = net.se_256(feat_16, feat_256)
#     if im_size==256:
#         return net.to_big(feat_256)
    
#     feat_512 = net.feat_512(feat_256)
#     feat_512 = net.se_512(feat_32, feat_512)
#     if im_size==512:
#         return net.to_big(feat_512)
    
#     feat_1024 = net.feat_1024(feat_512)
#     return net.to_big(feat_1024)

# feat_8_a, feat_16_a, feat_32_a, feat_64_a = get_early_features(netG, z)
# images_a = get_late_features(netG, 256, feat_64_a, feat_8_a, feat_16_a, feat_32_a)
# images_a = (images_a + 1) /2
# vutils.save_image(images_a, "TEST.png")



with torch.no_grad():
    generated_images = netG(z)[0]

generated_images = (generated_images + 1) / 2  # 轉換到 [0,1] 範圍
for i, img in enumerate(generated_images):
    vutils.save_image(img, f"generated_{i}.png")

print("✅ 圖片已生成並儲存！")
