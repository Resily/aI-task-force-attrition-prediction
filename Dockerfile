FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Jupyter用のカーネル設定
RUN python -m ipykernel install --user --name "turnover-prediction" --display-name "turnover-prediction"

# カーネルリストを表示して確認
RUN jupyter kernelspec list

# ディレクトリ作成
RUN mkdir -p src/data/raw/predict \
    src/data/processed/predict \
    src/data/processed/train \
    src/model/randomForest/param \
    src/model/randomForest/result

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

EXPOSE 8888