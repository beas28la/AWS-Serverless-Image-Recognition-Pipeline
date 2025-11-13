# AWS Serverless Image Recognition Pipeline

這是一個基於 AWS 的無伺服器圖像識別管道專案。

## 專案結構

```
AWS-Serverless-Image-Recognition-Pipeline/
├── onprem-ml-baseline/     # On-premise ML baseline 實作
│   ├── app.py              # Flask 應用程式主文件
│   ├── requirements.txt    # Python 依賴套件
│   ├── env.example         # 環境變數範例
│   ├── test_endpoints.py   # API 端點測試
│   └── README.md          # 詳細說明文件
```

## 開始使用

### 前置需求

- Python 3.8+
- pip

### 安裝

1. 克隆此儲存庫：
```bash
git clone https://github.com/YOUR_USERNAME/AWS-Serverless-Image-Recognition-Pipeline.git
cd AWS-Serverless-Image-Recognition-Pipeline
```

2. 安裝依賴套件：
```bash
cd onprem-ml-baseline
pip install -r requirements.txt
```

3. 設置環境變數：
```bash
cp env.example .env
# 編輯 .env 文件並填入您的配置
```

4. 運行應用程式：
```bash
python app.py
```

## 授權

MIT License

