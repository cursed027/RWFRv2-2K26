# models/adaface.py
import torch
import cv2
import numpy as np
from adaface import net



class AdaFaceModel:
    def __init__(self, device="cuda", architecture="ir_50"):
        self.device = device

        # Build model
        self.model = net.build_model(architecture)

        # Load pretrained weights (exactly like inference.py)
        ckpt = torch.load(
            "/media/admin1/DL/MILAN/3StageRWFR/repo/pretrained/adaface_ir50_ms1mv2.ckpt",
            map_location=device
        )["state_dict"]

        model_statedict = {
            k[6:]: v for k, v in ckpt.items()
            if k.startswith("model.")
        }

        self.model.load_state_dict(model_statedict)
        self.model.eval().to(device)

    def preprocess(self, img_bgr):
        """
        Input: aligned face, BGR, uint8, any size
        Output: torch tensor (1,3,112,112)
        """
        img = cv2.resize(img_bgr, (112, 112))
        img = img.astype(np.float32)

        # BGR, normalized to [-1, 1]
        img = ((img / 255.0) - 0.5) / 0.5
        img = img.transpose(2, 0, 1)

        return torch.from_numpy(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def __call__(self, img_bgr):
        inp = self.preprocess(img_bgr)
        feature, norm = self.model(inp)
        feature = feature.squeeze(0)
        feature = feature / torch.norm(feature, 2)
        return feature.cpu().numpy()
