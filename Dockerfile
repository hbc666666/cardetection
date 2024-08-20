# 基础镜像：选择适合的基础镜像
FROM nvidia/cuda:11.7.1-base-ubuntu18.04

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

# 设置工作目录
WORKDIR /workspace

# 复制模型、脚本和其他文件到容器中
COPY run.py /workspace/run.py
COPY best.pt /workspace/best.pt  # 替换为实际的模型文件路径

# 安装 Python 依赖
RUN pip3 install --upgrade pip
RUN pip3 install ultralytics opencv-python-headless lxml
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN pip3 install -r requirements.txt

# 定义容器启动时的默认命令
CMD ["python3", "run.py", "/input_path", "/output_path"]
