import os
import io
import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import timm

import streamlit as st
import gdown

# --------------------------------------------------
# 0) Paths and files
# --------------------------------------------------
# این‌ها باید در خود ریپو باشند
LABELS_CSV = "wash_labels.csv"     # already in repo
HEADER_IMG = "ai.jpg"              # already in repo

# نام فایل مدل روی سرور
CKPT_PATH = "best_model2_wash.pt"

# لینک مستقیم مدل روی گوگل‌درایو
# (از لینک: https://drive.google.com/file/d/15H1VcNrSRhjMA0eA4XxP6R5WFQwZZDf3/view?usp=drive_link)
MODEL_URL = "https://drive.google.com/uc?id=15H1VcNrSRhjMA0eA4XxP6R5WFQwZZDf3"

# اگر مدل روی سرور نبود، دانلودش کن
if not os.path.exists(CKPT_PATH):
    gdown.download(MODEL_URL, CKPT_PATH, quiet=False)

# --------------------------------------------------
# 1) Load label metadata
# --------------------------------------------------
df_all = pd.read_csv(LABELS_CSV)

df_all["color_label"] = df_all["color_label"].astype(int)
df_all["fabric_label"] = df_all["fabric_label"].astype(int)
df_all["wash_cycle_label"] = df_all["wash_cycle_label"].astype(int)

num_color_classes = df_all["color_label"].nunique()
num_fabric_classes = df_all["fabric_label"].nunique()
num_wash_classes = df_all["wash_cycle_label"].nunique()

# map از عدد به اسم کلاس
color_map = (
    df_all[["color_label", "color_group"]]
    .drop_duplicates("color_label")
    .set_index("color_label")["color_group"]
    .to_dict()
)

fabric_map = (
    df_all[["fabric_label", "fabric_group"]]
    .drop_duplicates("fabric_label")
    .set_index("fabric_label")["fabric_group"]
    .to_dict()
)

wash_map = (
    df_all[["wash_cycle_label", "wash_cycle"]]
    .drop_duplicates("wash_cycle_label")
    .set_index("wash_cycle_label")["wash_cycle"]
    .to_dict()
)

# توضیح کامل برنامه‌ی شستشو
wash_full_description = {
    "delicate": (
        "Delicate – Gentle wash, cold water, low spin "
        "(ideal for silk, lace, and sensitive fabrics)."
    ),
    "normal": (
        "Normal – Standard wash, 40°C warm water, medium spin "
        "(suitable for cotton and everyday clothes)."
    ),
    "heavy": (
        "Heavy – Deep clean, 60°C hot water, high spin "
        "(good for jeans, towels, and sportswear)."
    ),
    "quick": (
        "Quick – Rapid wash, 30°C cool water, medium spin "
        "(for lightly-soiled garments)."
    ),
    "wool": (
        "Wool – Special wool cycle, 30°C cold water, ultra-low spin "
        "(prevents shrinkage and protects fibers)."
    ),
}

# --------------------------------------------------
# 2) Model definition (must match training)
# --------------------------------------------------
BACKBONE_NAME = "convnext_tiny"


class WashMultiTaskConvNeXt(nn.Module):
    def __init__(self, num_color, num_fabric, num_wash):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features

        self.head_color = nn.Linear(feat_dim, num_color)
        self.head_fabric = nn.Linear(feat_dim, num_fabric)
        self.head_wash_cycle = nn.Linear(feat_dim, num_wash)  # name matches checkpoint

    def forward(self, x):
        feat = self.backbone(x)
        logits_color = self.head_color(feat)
        logits_fabric = self.head_fabric(feat)
        logits_wash = self.head_wash_cycle(feat)
        return logits_color, logits_fabric, logits_wash


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = WashMultiTaskConvNeXt(
    num_color=num_color_classes,
    num_fabric=num_fabric_classes,
    num_wash=num_wash_classes,
).to(device)

state_dict = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# --------------------------------------------------
# 3) Preprocessing (بدون transforms.ToTensor)
# --------------------------------------------------
IMG_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(pil_img: Image.Image) -> torch.Tensor:
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE))

    x = np.array(pil_img).astype("float32") / 255.0  # [H, W, 3]
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return x


# --------------------------------------------------
# 4) Single-image prediction helper
# --------------------------------------------------
def predict_single_image(pil_img: Image.Image):
    x = preprocess(pil_img).to(device)

    with torch.no_grad():
        logits_c, logits_f, logits_w = model(x)
        pc = logits_c.argmax(1).item()
        pf = logits_f.argmax(1).item()
        pw = logits_w.argmax(1).item()

    color_name = color_map.get(pc, f"class {pc}")
    fabric_name = fabric_map.get(pf, f"class {pf}")
    wash_key = wash_map.get(pw, f"class {pw}")
    wash_text = wash_full_description.get(wash_key, str(wash_key))

    return {
        "color": color_name,
        "fabric": fabric_name,
        "wash_key": wash_key,
        "wash_text": wash_text,
    }


# --------------------------------------------------
# 5) Streamlit UI
# --------------------------------------------------
def main():
    st.set_page_config(
        page_title="AI Laundry Sorter",
        page_icon="🧺",
        layout="centered",
    )

    if os.path.exists(HEADER_IMG):
        st.image(HEADER_IMG, use_container_width=True)

    st.title("AI Laundry Sorter")
    st.caption(
        "Deep Learning–powered multi-task model that predicts color, fabric, "
        "and the recommended washing program for each garment."
    )

    uploaded_file = st.file_uploader(
        "Upload a clothing image (from your dataset or any new garment)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Please upload a garment image to see the recommended wash settings.")
        return

    # read image
    pil_img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input garment")
        st.image(pil_img, use_container_width=True)

    # prediction
    result = predict_single_image(pil_img)

    with col2:
        st.subheader("Recommended settings")
        st.markdown(f"**Color group:** {result['color']}")
        st.markdown(f"**Fabric group:** {result['fabric']}")
        st.markdown(f"**Wash program:** {result['wash_text']}")

    # Optional simple log (no need برای پروژه هم خیلی خوبه)
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    st.caption(
        f"Prediction generated at {ts}. "
        "Model: ConvNeXt-tiny multi-task classifier (color, fabric, wash cycle)."
    )


if __name__ == "__main__":
    main()
