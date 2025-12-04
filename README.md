## 心臟肌肉影像分割專案說明

本專案實作 3D Residual U-Net，用於 AICUP 心臟肌肉影像分割。主要流程分為兩個訓練版本（`train_v1.ipynb`, `train_v2.ipynb`）以及一個推論與集成版本（`inference.ipynb`）。

---

## 環境與依賴

- **主要套件**：`torch`, `torchvision`, `nibabel`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`, `scipy`
- 建議於 **Google Colab + GPU** 執行（程式中已包含 `google.colab` 掛載雲端硬碟與路徑設定）
- 資料格式：3D NIfTI (`.nii` / `.nii.gz`)，大小約 \(512 \times 512 \times 3xx\)

---

## 資料結構建議

Google Drive 範例結構（可依實際情況調整，但需同步修改 notebook 中路徑）：

- **`MyDrive/aicup_data/training_image`**：訓練影像（如 `patient0001.nii.gz`）
- **`MyDrive/aicup_data/training_label`**：對應標籤（如 `patient0001_gt.nii.gz`）
- **`MyDrive/aicup_data/testing_image`**：競賽測試影像
- **`MyDrive/aicup_data/aicup_results`**：訓練產生的最佳模型與壓縮後提交結果

---

## train_v1.ipynb：Residual U-Net + 預訓練微調版本

- **目的**：載入既有最佳模型權重，再以 **patch-based 訓練策略**做微調（fine-tune）。
- **模型**：
  - 架構：`UNet3D` + `ResConv3DBlock`（encoder / decoder 全部採殘差卷積）
  - 輸入通道：1，輸出類別：4（0: 背景, 1–3: 心臟組織）
  - `base_channels=32`，約 2,293 萬參數
- **資料處理**：
  - 先將整個 volume 降採樣至 `320 × 320 × 160`，並正規化到 `[0, 1]`
  - 使用 `PatchBasedHeartDataset`：
    - Patch 大小：`(112, 112, 112)`
    - 每個 volume 抽取 4 個 patch
    - 70% 機率從「前景區域」抽樣，提升心臟區域訓練比例
- **資料增強**（訓練集啟用）：
  - 隨機旋轉（±15°）、左右 / 前後翻轉
  - 高斯雜訊、亮度 / 對比度隨機縮放
  - 若環境有安裝 `torchio`，額外加入 3D elastic deformation
- **損失函數**：`CombinedLoss = 0.4 * CE + 0.3 * Dice + 0.3 * Boundary`
  - CE 使用類別權重：`[0.5, 1.5, 1.5, 1.5]` 處理類別不平衡
  - Boundary loss 著重邊界區域誤差
- **訓練設定**：
  - optimizer: `Adam(lr=1e-4, weight_decay=1e-4)`
  - scheduler: `ReduceLROnPlateau`（以 val loss 調整 LR）
  - epochs：150
  - 以 **(Dice + IoU) / 2 的 Combined Score** 選擇最佳 checkpoint（`checkpoints/best_model.pth`）
- **輸出**：
  - 訓練曲線圖：`training_curves.png`
  - 驗證視覺化：`validation_results.png`
  - 最佳模型會同步複製到 `MyDrive/aicup_data/aicup_results/best_model_*.pth`

**適用情境**：已有舊版 3D U-Net 權重，希望在新 loss / 新資料增強下再微調一次模型。

---

## train_v2.ipynb：Residual U-Net 強化版（更大模型 + 強增強 + 梯度累積）

- **目的**：從頭訓練更大容量模型，搭配更強的 3D 資料增強與梯度累積，以提升表現。
- **模型**：
  - 架構：`UNet3D` + `ResidualConv3DBlock`
  - `base_channels=48`，參數量約 5,158 萬
- **資料處理**：
  - 將完整 volume 降採樣至 `384 × 384 × 192`
  - 同樣使用 `PatchBasedHeartDataset`，設定：
    - Patch 大小：`(112, 112, 112)`
    - 每個 volume 4 個 patches
    - 70% 前景導向抽樣
- **資料增強**（強化版）：
  - 旋轉、左右 / 前後翻轉
  - **3D Elastic Deformation**（自寫實作，使用高斯平滑位移場）
  - 高斯雜訊、亮度 / 對比度調整
- **損失函數**：`CombinedLoss = 0.2 * CE + 0.45 * Dice + 0.35 * Boundary`
  - 仍使用類別權重 `[0.5, 1.5, 1.5, 1.5]`
  - 整體更偏向區域重疊與邊界表現
- **訓練技巧**：
  - **梯度累積**：`BATCH_SIZE=4`, `GRAD_ACCUM_STEPS=4`
    - 有效 batch size ≈ 16，但單次 forward 仍只吃 4 個 patches，較省顯存
  - 其他設定與 `train_v1` 類似：
    - `Adam(lr=1e-4, weight_decay=1e-4)`
    - `ReduceLROnPlateau`
    - epochs=150，使用 Combined Score 選最優模型
- **輸出**：
  - `checkpoints/best_model.pth`
  - 同步備份到 `MyDrive/aicup_data/aicup_results/best_model_*.pth`

**適用情境**：希望單一模型表現更強、可以接受較長訓練時間與較大顯存佔用。

---

## inference.ipynb：Weighted Ensemble 推論 + 後處理

- **目的**：將兩個訓練完成的模型做 **Soft Voting Ensemble**，在測試集產生最終分割結果並壓縮成提交檔。
- **使用模型**（以 notebook 內目前設定為例，可依實際路徑修改）：
  - Model 1（ver9）：`base_channels=32`，輸入解析度 `320×320×160`
  - Model 2（ver11）：`base_channels=48`，輸入解析度 `384×384×192`
  - 權重皆從 `MyDrive/aicup_data/aicup_results/best_model_*.pth` 載入
- **推論步驟**：
  1. 掛載 Google Drive、讀取 `testing_image` 中的 NIfTI 影像。
  2. 影像做 1–99 百分位正規化到 `[0, 1]`。
  3. 對每張影像：
     - 依各自 target size 重採樣成 Model 1 / Model 2 需要的解析度。
     - 使用 **3D sliding window**（patch 大小 `160×160×160`，overlap=0.8）跑整張 volume。
     - 選配 **TTA**（預設：Model1 開 TTA，Model2 關；可在程式內切換），TTA 為原圖 + 水平翻轉。
     - 將兩模型的機率輸出重採樣回原始解析度。
  4. **Ensemble**：
     - `probs = 0.6 * probs_model1 + 0.4 * probs_model2`
     - 對 channel 方向取 argmax 得到最終 label。
  5. **後處理**：
     - 對類別 1, 2, 3 做 3D Largest Connected Component（LCC），只保留最大連通元件，可設定 `min_size` 門檻（預設 300）。
  6. 儲存結果到 `testing_output` 資料夾，檔名為 `patientXXXX_pred.nii.gz`。
- **效能與 I/O 優化**：
  - 減少不必要的 array 複製與型別轉換
  - `ndimage.zoom` 關閉 prefilter 加速重採樣
  - 定期 `torch.cuda.empty_cache()` 降低 GPU 記憶體碎片
  - 使用 `zipfile` 以中等壓縮等級打包全部預測檔案到 `MyDrive/aicup_data/aicup_results/testing_output_ensemble_*.zip`

**適用情境**：產生最終競賽提交結果，整批處理測試集 50 例並自動壓縮。

---