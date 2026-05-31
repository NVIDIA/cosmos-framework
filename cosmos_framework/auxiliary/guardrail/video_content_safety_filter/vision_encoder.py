# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch
from PIL import Image
from transformers import SiglipModel, SiglipProcessor


class SigLIPEncoder(torch.nn.Module):
    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",  # noqa: B008
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        model_id = "google/siglip-so400m-patch14-384"
        self.model = SiglipModel.from_pretrained(model_id)
        self.processor = SiglipProcessor.from_pretrained(model_id)
        self.model.to(self.device, dtype=self.dtype).eval()

    @torch.inference_mode()
    def encode_image(self, input_img: Image.Image) -> torch.Tensor:
        """Encode an image into a feature vector."""
        with torch.no_grad():
            inputs = self.processor(images=input_img, return_tensors="pt").to(self.device, dtype=self.dtype)
            image_features = self.model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
