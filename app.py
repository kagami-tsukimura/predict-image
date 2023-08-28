import time

import matplotlib.pyplot as plt
import streamlit as st
import torch
from googletrans import Translator
from PIL import Image
from torchvision import models, transforms


def predict(img):
    """
    与えられた画像に対して予測を行い、確信度の高いクラスとその確信度を返す。

    Parameters:
        img (PIL.Image): 予測対象の画像。

    Returns:
        List[Tuple[str, float]]: クラス名とその確信度のリスト。
    """
    preprocessed_image = preprocess_image(img)
    img_tensor = torch.unsqueeze(preprocessed_image, 0)

    model = models.resnet101(pretrained=True)
    model.eval()
    output = model(img_tensor)

    output_prob = torch.nn.functional.softmax(torch.squeeze(output))
    sorted_prob, sorted_indices = torch.sort(output_prob, descending=True)
    CLASSES_FILE = "classes/imagenet_classes.txt"
    with open(CLASSES_FILE) as f:
        classes = [line.strip() for line in f.readlines()]
    return [
        (classes[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)
    ]


def preprocess_image(img):
    """
    画像を前処理して、モデルへの入力形式に変換する。

    Parameters:
        image (PIL.Image): 前処理対象の画像。

    Returns:
        torch.Tensor: 前処理された画像のテンソル。
    """
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    preprocessed_image = transform(img)
    return preprocessed_image


def main():
    """
    画面表示する処理。
    """

    st.sidebar.title("画像認識アプリ")

    st.sidebar.write("**ResNet**を使って何の画像かを判定します。")
    st.sidebar.write("")

    SOURCE_DICT = {"upload": "画像をアップロード", "camera": "カメラで撮影"}

    img_source = st.sidebar.radio("画像のソースを選択してください。", tuple(SOURCE_DICT.values()))
    if img_source == SOURCE_DICT["upload"]:
        img_file = st.file_uploader("画像を選択してください。", type=["png", "jpg"])
    elif img_source == SOURCE_DICT["camera"]:
        img_file = st.camera_input("カメラで撮影")

    if img_file:
        with st.spinner("判定中..."):
            img = Image.open(img_file)
            st.image(img, caption="対象の画像", width=480)
            st.write("")

            results = predict(img)
            translator = Translator()
            predict_result_ja = translator.translate(
                results[0][0], src="en", dest="ja"
            ).text

            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"判定結果: {predict_result_ja}")
                # 上位5クラスまで返す
                N_TOP_5 = 5
                results_ja = []
                for i, result in enumerate(results[:N_TOP_5]):
                    results_ja.append(
                        translator.translate(result[0], src="en", dest="ja").text
                    )
                    st.markdown(
                        f"{i+1}. {str(round(result[1] * 100, 2))}%の確率で{results_ja[i]}です。"
                    )

            with col2:
                pie_labels = [result_ja for result_ja in results_ja[:N_TOP_5]]
                pie_labels.append("others")
                pie_probs = [result[1] for result in results[:N_TOP_5]]
                pie_probs.append(sum([result[1] for result in results[N_TOP_5:]]))
                result_dict = {k: v for k, v in zip(pie_labels, pie_probs)}
                # NOTE: 3%以上のみグラフに表示して残りはothers
                filtered_result_dict = {
                    k: v for k, v in result_dict.items() if v >= 0.03
                }

                plt.rcParams["font.family"] = "Noto Sans CJK JP"
                fig, ax = plt.subplots()
                wedgeprops = {"width": 0.3, "edgecolor": "white"}
                textprops = {"fontsize": 12}
                ax.pie(
                    list(filtered_result_dict.values()),
                    labels=list(filtered_result_dict.keys()),
                    counterclock=False,
                    startangle=90,
                    textprops=textprops,
                    autopct="%.2f",
                    wedgeprops=wedgeprops,
                )
                st.pyplot(fig)
            st.toast("完了！", icon="🎉")
            time.sleep(1)
            st.toast("TIPS: 他の画像でも是非試してみてくださいね！", icon="🐧")


if __name__ == "__main__":
    main()
