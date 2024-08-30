# 基础镜像：选择适合的基础镜像
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu18.04

# 设置维护者信息
LABEL maintainer="Your Name <your.email@example.com>"

# 更新系统并安装必要的软件包
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# 设置工作目录
WORKDIR /workspace

# 复制模型、脚本和其他文件到容器中
COPY . /workspace
#COPY run.py /workspace/run.py
#COPY best.pt /workspace/best.pt  # 替换为实际的模型文件路径

# 安装 Python 依赖
#RUN pip3 install --upgrade pip
RUN pip3 install -U pip
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --default-timeout=1000 opencv-python-headless==4.6.0.66 --only-binary=:all:
#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN --mount=type=cache,target=/root/.cache/pip pip3 install -r requirements.txt


# 定义容器启动时的默认命令
CMD ["python", "run.py", "/input_path", "/output_path"]
