# ベースイメージ：StyleGAN2公式環境に寄せる
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo

ARG UID=3059
ARG GID=1000

# 追加で欲しいパッケージ（最低限）
RUN apt-get update && \
    apt-get install -y git python3 python3-pip wget bash && \
    rm -rf /var/lib/apt/lists/*


# 画像系ライブラリ(Pillowビルドに必要)
RUN apt-get update && \
    apt-get install -y zlib1g-dev libjpeg-dev libpng-dev && \
    rm -rf /var/lib/apt/lists/*


# Pythonパッケージ（StyleGAN2 TF版に必要）
RUN pip3 install --no-cache-dir \
    numpy==1.16.3 \
    scipy \
    pillow \
    matplotlib \
    imageio \
    tqdm


# TensorFlow 1.14 用に protobuf を古い互換版に固定してから入れる
RUN pip3 install "protobuf==3.19.6" && \
    pip3 install "tensorflow-gpu==1.14.0"
    
# ユーザー作成
RUN groupadd -g $GID riri && \
   useradd -m -u $UID -g $GID riri


# ワークスペースの作成
RUN mkdir -p /workspace && chown -R riri:riri /workspace
WORKDIR /workspace

USER riri

CMD ["/bin/bash"]