import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.models import resnet34


# 重みは何度も読み込むのが無駄なので cache するように decorator で指定する
@st.cache
def load_model() -> torch.nn.Module:
    model = resnet34(pretrained=True)
    return model


def img2tensor(img: Image.Image) -> torch.Tensor:
    transformer = T.Compose([
        T.Resize(300),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    x = transformer(img)
    x = x.unsqueeze(0)
    return x


def annotate_labels(output: torch.Tensor) -> pd.DataFrame:
    # imagenet のラベルを読み込み. read_csv は url 指定できる
    label_df = pd.read_csv("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
    # 出力を確率へ変換
    prob = F.softmax(output[0], dim=0)
    pred_df = pd.DataFrame(prob)

    # ラベル情報と, 予測確率を対応させて, 確率の高いものが上に来るようにする
    pred_df = pd.concat([label_df, pred_df], axis=1, ignore_index=True)
    pred_df.columns = ["label", "prob"]
    pred_df = pred_df.set_index("label")
    pred_df = pred_df.sort_values("prob", ascending=False)
    return pred_df


def main():
    # 学習済みのモデルの読み込み
    model = load_model()

    # 推論対象の画像を読んできて tensor へ変換
    img = Image.open("/path/to/img.jpeg")
    tensor = img2tensor(img)

    # 推論実行
    with torch.no_grad():
        model.eval()
        output = model(tensor)

    # imagenet のラベル付
    pred_df = annotate_labels(output)
    pred_df.to_csv("path/to/output.csv")


if __name__ == "__main__":
    main()
