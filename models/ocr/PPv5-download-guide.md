# PaddleOCR 模型轉換 ONNX 指南

本文件說明如何使用 **PaddleX** + **Paddle2ONNX**，將 HuggingFace 上的 PaddleOCR v5 模型轉換為 ONNX，並配置於 PDF 辨識流程中。

---

## 1. 建立環境 (建議開新專案)

```bash
cd ~/WinDiskD/Code/temp/paddle2onnx
pipenv install --python=3.11.4
```

---

## 2. 安裝必要套件

### 安裝 PaddleX 與 Paddle2ONNX

```bash
pipenv install paddlex
pipenv run paddlex --install paddle2onnx
```

### 安裝 PaddlePaddle

```bash
pipenv install paddlepaddle
```

> ⚠️ 根據硬體環境選擇安裝 **CPU** 或 **GPU** 版本的 PaddlePaddle  
> - CPU: `pipenv install paddlepaddle`  
> - GPU (CUDA): `pipenv install paddlepaddle-gpu`

---

## 3. 安裝 Git LFS

HuggingFace 上的大模型檔案需要 **Git LFS** 支援。

```bash
sudo apt install git-lfs
git lfs install
```

---

## 4. 下載模型 (HuggingFace)

以 **PP-OCRv5** 為例：

```bash
git clone https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det
git clone https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec
```

下載完成後，資料夾中會包含：
- `inference.pdmodel`  
- `inference.pdiparams`  
- `inference.yml`

---

## 5. 轉換為 ONNX

### 範例：Detection 模型

```bash
pipenv run paddlex --paddle2onnx \
  --paddle_model_dir ./PP-OCRv5_server_det \
  --onnx_model_dir ./onnx_model/PP-OCRv5_server_det \
  --opset_version 11
```

### 範例：Recognition 模型

```bash
pipenv run paddlex --paddle2onnx \
  --paddle_model_dir ./PP-OCRv5_server_rec \
  --onnx_model_dir ./onnx_model/PP-OCRv5_server_rec \
  --opset_version 11
```

---

## 6. 輸出結果

轉換完成後，輸出目錄會包含：

- `inference.onnx`  
- `inference.yml`  

例如：

```
onnx_model/
 ├── PP-OCRv5_server_det/
 │    ├── inference.onnx
 │    └── inference.yml
 └── PP-OCRv5_server_rec/
      ├── inference.onnx
      └── inference.yml
```

---

## 7. 設定 rec 模型所需字典檔

- 將 rec 模型中，inference.yml 下的 PostProcess.character_dict 另存成以下格式的 txt
- 命名為 chars.txt 放在與 inference.onnx 相同的的路徑下

```txt
　
一
乙
二
十
丁
厂
七
卜
八
人
入
儿
匕
几
...
```

---

## 8. 設定 RapidOCR 路徑

在 src/ingestion/file_loaders/pdf.py 修改 `RapidOcrOptions`

```python
from huggingface_hub import snapshot_download
from docling.datamodel.pipeline_options import RapidOcrOptions

download_path = snapshot_download(repo_id="SWHL/RapidOCR")
cls_model_path = os.path.join(download_path, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx")
ocr_options = RapidOcrOptions(lang=["english", "chinese", "japanese"],  # Docling 說這參數沒用
                              force_full_page_ocr=False,
                              det_model_path=".../onnx_model/PP-OCRv5_server_det/inference.onnx",
                              rec_model_path=".../onnx_model/PP-OCRv5_server_rec/inference.onnx",
                              cls_model_path=cls_model_path,
                              rec_keys_path=".../onnx_model/PP-OCRv5_server_rec/chars.txt"
                              )
```

---

## 9. 測試

- 預設 `--opset_version` 為 7，但 Paddle2ONNX 會自動升級至相容版本 (如 10, 11)  
- 若遇到 `ModuleNotFoundError: paddle` → 確認已安裝 `paddlepaddle` 或 `paddlepaddle-gpu`  
- HuggingFace 模型必須用 **git-lfs** 拉取，否則 `.pdmodel` 會是空檔案  

---

---

## 10. 注意事項

- 預設 `--opset_version` 為 7，但 Paddle2ONNX 會自動升級至相容版本 (如 10, 11)  
- 若遇到 `ModuleNotFoundError: paddle` → 確認已安裝 `paddlepaddle` 或 `paddlepaddle-gpu`  
- HuggingFace 模型必須用 **git-lfs** 拉取，否則 `.pdmodel` 會是空檔案  
- 目前本機專案位置在： `/Code/temp/paddle2onnx/Pipfile.lock`
- GPU 消耗約落在 1G，如果不使用 GPU 辨識，將套建中的 onnxruntime-gpu 解除安裝即可

---
