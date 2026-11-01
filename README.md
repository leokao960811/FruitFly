# 電子鼻結合果蠅嗅覺神經概念之氣味辨識與實體 AI 回應系統

本專題以 BME688 與 SGP41 電子鼻感測資料為輸入，參考果蠅（*Drosophila*）嗅覺神經迴路，逐步建立氣味特徵編碼、神經元連結分析與腦部活動視覺化流程。

目前已完成的工作是：從 MaleCNS v1.0 神經元註解資料篩選候選嗅覺相關神經元，再使用 `fly-connectome-template` 內的腦圖譜 ID 清單確認哪些神經元可以被範本顯示。

> **重要定位**：`fly-connectome-template` 是腦部資料視覺化與模型整合範本，不是已經完成的嗅覺神經模擬器。感測器特徵如何轉成神經活動、如何建立神經網路，以及 AI／實體裝置如何回應，都需要由本專題另外實作。

---

## 1. 專題目前流程

```text
BME688 + SGP41 感測資料
          │
          ▼
資料前處理與氣味特徵提取
          │
          ▼
虛擬嗅覺受體（Virtual ORN）編碼       ← 待實作
          │
          ▼
MaleCNS 神經元與突觸連結分析           ← 神經元篩選已完成；連結分析待進行
          │
          ▼
神經元活動模型                         ← 待實作
          │
          ▼
神經元活動 JSON
          │
          ▼
fly-connectome-template 腦部視覺化
          │
          ▼
AI 判斷與實體裝置回應                  ← 待整合
```

## 2. 資料來源

### 2.1 MaleCNS v1.0 神經元註解資料

從 [MaleCNS 官方下載頁面](https://male-cns.janelia.org/download/) 下載以下 Feather 檔案：

```text
body-annotations-male-cns-v1.0-minconf-0.5.feather
```

此資料包含神經元的 `bodyId`、`type`、`class`、`instance`、`somaLocation`、`status` 等註解資訊。實際欄位請以下載版本中的資料為準。

### 2.2 fly-connectome-template

專案來源：[cobanov/fly-connectome-template](https://github.com/cobanov/fly-connectome-template)

下載 ZIP 或使用 Git clone，並保留以下資料：

```text
fly-connectome-template/
└── public/
    └── data/
        └── brain-atlas/
            ├── ids.bin
            ├── groups.bin
            └── positions.bin
```

本階段使用：

| 檔案 | 用途 |
|---|---|
| `ids.bin` | 腦圖譜中的神經元 `bodyId`，格式為 little-endian uint32 |
| `groups.bin` | 與 `ids.bin` 逐列對應的顯示群組，格式為 uint8 |
| `positions.bin` | 神經元空間座標；本階段尚未使用 |

範本目前的可顯示神經元篩選條件為 `groups < 3`。**`ids.bin` 與 `groups.bin` 必須依相同索引逐列對應，不可分開排序。**

### 2.3 電子鼻資料

本專題目前的硬體資料來自：

- BME688：氣體電阻（Gas resistance）、溫度、濕度、氣壓等。
- SGP41：VOC 與 NOx 原始訊號（SRAW_VOC、SRAW_NOX）。

請留意：SGP41 的原始訊號不是 VOC Index／NOx Index，也不是直接的氣體濃度。若後續要使用 Gas Index Algorithm，應依 Sensirion 官方文件處理取樣頻率與演算法輸入。

---

## 3. 環境準備

建議使用 Python 3 環境。安裝目前資料處理會用到的套件：

```bash
pip install pandas pyarrow numpy
```

若要使用 neuPrint API 查詢神經元連結，後續再安裝：

```bash
pip install neuprint-python
```

### 建議資料夾結構

以下是本 README 範例程式所使用的相對路徑。若實際資料夾名稱不同，請修改程式最前面的路徑設定：

```text
project-root/
├── README.md
├── search_neurons.py
├── data/
│   └── body-annotations-male-cns-v1.0-minconf-0.5.feather
├── fly-connectome-template/
│   └── public/
│       └── data/
│           └── brain-atlas/
│               ├── ids.bin
│               ├── groups.bin
│               └── positions.bin
└── output/
```

原始 CSV、Feather 與大型資料檔不一定需要放進 Git repository。請依團隊需求決定是否納入版本控制；API Token、密碼等憑證絕對不要提交到 GitHub。

---

## 4. 篩選嗅覺相關候選神經元

目前使用 MaleCNS 註解資料中的 `class` 欄位進行精確篩選，而不是只靠模糊關鍵字搜尋。

本階段使用的候選分類：

| `class` | 說明 | 本次篩選數量 |
|---|---|---:|
| `ALIN` | Antennal Lobe Interneuron，觸角葉局部神經元 | 24 |
| `ALPN` | Antennal Lobe Projection Neuron，觸角葉投射神經元 | 686 |
| `MBON` | Mushroom Body Output Neuron，蘑菇體輸出神經元 | 97 |
| `DAN` | Dopaminergic Neuron，多巴胺神經元 | 340 |
| **合計** |  | **1,147** |

注意：這些是依 `class` 選出的候選分類，不代表每一顆 DAN 或 MBON 都已確認屬於特定嗅覺路徑。後續仍須透過神經元連結資料確認實際路徑。Kenyon Cells（KC）尚未納入本次四類篩選，需另外檢查註解資料中的命名與分類。

### Python 程式

將以下內容存成 `search_neurons.py`，並依照實際資料夾位置修改路徑：

```python
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# 路徑設定：依照實際專案資料夾修改
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

ANNOTATION_FILE = (
    PROJECT_DIR
    / "data"
    / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)

ATLAS_DIR = (
    PROJECT_DIR
    / "fly-connectome-template"
    / "public"
    / "data"
    / "brain-atlas"
)

OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. 讀取 MaleCNS 神經元註解資料
# --------------------------------------------------
neurons = pd.read_feather(ANNOTATION_FILE)

required_columns = {"bodyId", "class"}
missing_columns = required_columns - set(neurons.columns)

if missing_columns:
    raise ValueError(
        f"註解資料缺少必要欄位：{sorted(missing_columns)}"
    )

# 只選取本階段研究的候選分類
target_classes = ["ALIN", "ALPN", "MBON", "DAN"]

olfactory_neurons = neurons[
    neurons["class"].isin(target_classes)
].copy()

# bodyId 必須可轉換為整數
olfactory_neurons["bodyId"] = pd.to_numeric(
    olfactory_neurons["bodyId"],
    errors="raise"
).astype("int64")

print("候選神經元總數：", len(olfactory_neurons))
print("各分類數量：")
print(olfactory_neurons["class"].value_counts())

# 儲存尚未與範本比對的候選清單
olfactory_neurons.to_csv(
    OUTPUT_DIR / "olfactory_neurons.csv",
    index=False,
    encoding="utf-8-sig"
)

# --------------------------------------------------
# 2. 讀取 fly-connectome-template 腦圖譜資料
# --------------------------------------------------
ids_path = ATLAS_DIR / "ids.bin"
groups_path = ATLAS_DIR / "groups.bin"

if not ids_path.exists() or not groups_path.exists():
    raise FileNotFoundError(
        "找不到 ids.bin 或 groups.bin，請檢查 ATLAS_DIR 路徑。"
    )

# ids.bin：little-endian uint32
ids = np.fromfile(ids_path, dtype="<u4")

# groups.bin：uint8
groups = np.fromfile(groups_path, dtype="u1")

if len(ids) != len(groups):
    raise ValueError(
        f"ids.bin 與 groups.bin 長度不一致："
        f"{len(ids)} vs {len(groups)}"
    )

# 範本規定 groups < 3 的神經元可顯示
visible_ids = set(
    ids[groups < 3].astype("int64").tolist()
)

all_atlas_ids = set(
    ids.astype("int64").tolist()
)

print("腦圖譜 ID 數量：", len(ids))
print("符合 groups < 3 的 ID 數量：", len(visible_ids))

# --------------------------------------------------
# 3. 以 bodyId 比對範本可顯示神經元
# --------------------------------------------------
olfactory_neurons["in_ids_bin"] = (
    olfactory_neurons["bodyId"].isin(all_atlas_ids)
)

olfactory_neurons["is_visible"] = (
    olfactory_neurons["bodyId"].isin(visible_ids)
)

# 產生統計表
summary = olfactory_neurons.groupby("class")["is_visible"].agg(
    total="count",
    visible="sum"
)

summary["not_visible"] = (
    summary["total"] - summary["visible"]
)

print("\n各類神經元比對結果：")
print(summary)

# 儲存所有候選神經元及比對狀態
olfactory_neurons.to_csv(
    OUTPUT_DIR / "olfactory_neurons_with_visibility.csv",
    index=False,
    encoding="utf-8-sig"
)

# --------------------------------------------------
# 4. 儲存可顯示清單與未納入清單
# --------------------------------------------------
visible_neurons = olfactory_neurons[
    olfactory_neurons["is_visible"]
].copy()

excluded_neurons = olfactory_neurons[
    ~olfactory_neurons["is_visible"]
].copy()

visible_neurons.to_csv(
    OUTPUT_DIR / "visible_olfactory_neurons.csv",
    index=False,
    encoding="utf-8-sig"
)

excluded_neurons.to_csv(
    OUTPUT_DIR / "excluded_olfactory_neurons.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n可顯示的候選神經元：", len(visible_neurons))
print("未納入範本的候選神經元：", len(excluded_neurons))
print("輸出資料夾：", OUTPUT_DIR)
```

執行：

```bash
python search_neurons.py
```

成功後會在 `output/` 產生：

| 檔案 | 內容 |
|---|---|
| `olfactory_neurons.csv` | 依 `class` 篩選出的全部候選神經元 |
| `olfactory_neurons_with_visibility.csv` | 候選神經元及其範本顯示狀態 |
| `visible_olfactory_neurons.csv` | 符合範本顯示條件的神經元 |
| `excluded_olfactory_neurons.csv` | 未能對應至範本顯示清單的神經元 |

---

## 5. 目前已完成的比對結果

本團隊目前使用的資料版本與執行結果如下：

### 5.1 腦圖譜資料讀取

| 項目 | 數量 |
|---|---:|
| `ids.bin` 神經元 ID 數量 | 140,024 |
| `groups.bin` 群組資料數量 | 140,024 |
| 符合 `groups < 3` 的神經元數量 | 124,289 |

`ids.bin` 與 `groups.bin` 長度相同，且已成功讀取。

### 5.2 候選神經元比對

| `class` | 候選數量 | 可顯示數量 | 未納入數量 |
|---|---:|---:|---:|
| ALIN | 24 | 24 | 0 |
| ALPN | 686 | 680 | 6 |
| DAN | 340 | 340 | 0 |
| MBON | 97 | 96 | 1 |
| **合計** | **1,147** | **1,140** | **7** |

保留比例：`1,140 / 1,147 ≈ 99.39%`。

### 5.3 未出現在 `ids.bin` 的 7 顆神經元

本次檢查確認以下 7 個 `bodyId` 不存在於範本的 `ids.bin`，因此無法由目前這份範本腦圖譜顯示：

| bodyId | type | class |
|---:|---|---|
| 10262 | V_ilPN | ALPN |
| 10973 | VP1m+VP5_ilPN | ALPN |
| 15313 | VA1d_adPN | ALPN |
| 139814 | VM6_lvPN | ALPN |
| 394860 | MBON25 | MBON |
| 512883 | DL2d_adPN | ALPN |
| 525105 | DL5_adPN | ALPN |

目前保留 1,140 顆作為後續分析候選清單。若資料集版本或範本版本更新，請重新執行比對，不要直接沿用舊統計數字。

---

## 6. 欄位與比對注意事項

1. **`bodyId` 是跨資料集對照的主要識別碼。** 不可使用 Pandas 左側的 DataFrame index 代替 `bodyId`。
2. `ids.bin` 與 `groups.bin` 的同一個索引代表同一顆神經元。不可各自排序後再配對。
3. 註解表中的 `group` 與範本中的 `groups.bin` 不是同一個欄位，不可混用。
4. `assignedOlHex1`、`assignedOlHex2` 出現 `NaN` 不會影響本階段的 `bodyId` 比對；它們不是本階段的篩選條件。
5. `somaLocation` 是細胞體座標，不是神經元 ID，也不能用來代替 `bodyId`。
6. `ALIN`、`ALPN`、`MBON`、`DAN` 是本階段的候選分類。是否屬於特定嗅覺路徑，仍需檢查上游／下游神經元及突觸連結。
7. 目前尚未把電子鼻的 VOC、NOx、Gas resistance 等數值映射到真實神經元，也尚未建立經生理驗證的神經活動模型。

---

## 7. 下一階段工作（尚未完成）

### 7.1 搜尋 Kenyon Cells（KC）

目前四類篩選尚未包含 KC。下一步應先檢查 MaleCNS 註解資料中的 `type`、`class`、`superclass`、`subclass`、`instance` 等欄位，確認 KC 的實際命名方式，再獨立篩選及進行 ID 比對。

### 7.2 取得神經元連結資料

可從 [MaleCNS 官方下載頁面](https://male-cns.janelia.org/download/) 取得 connectome weights，或使用 [neuPrint](https://neuprint.janelia.org/)／[neuprint-python 文件](https://connectome-neuprint.github.io/neuprint-python/docs/) 查詢特定神經元的連結。

API 查詢前需確認：

- 使用的 dataset 為 `male-cns:v1.0`。
- 查詢的 `bodyId` 屬於正確的 MaleCNS 資料版本。
- 區分 presynaptic（pre）與 postsynaptic（post）方向。
- 先用小型神經元子集合測試，再擴大查詢範圍。
- 連結權重是資料集中記錄的連結強度，不應直接宣稱等於生理神經傳導效率。

### 7.3 建立神經活動輸出

完成感測器特徵編碼與神經網路模型後，才能依照範本要求輸出神經元活動 JSON。範本要求使用 `MaleCNS_bodyId` 與正規化活動值，且只接受符合腦圖譜顯示範圍的 ID。

格式與限制請參考範本文件：[MODEL-INTEGRATION.md](https://github.com/cobanov/fly-connectome-template/blob/main/docs/MODEL-INTEGRATION.md)。

---

## 8. 參考連結

- [fly-connectome-template GitHub](https://github.com/cobanov/fly-connectome-template)
- [fly-connectome-template：模型整合文件](https://github.com/cobanov/fly-connectome-template/blob/main/docs/MODEL-INTEGRATION.md)
- [MaleCNS 官方下載頁面](https://male-cns.janelia.org/download/)
- [neuPrint 平台](https://neuprint.janelia.org/)
- [neuprint-python 文件](https://connectome-neuprint.github.io/neuprint-python/docs/)
- [Sensirion SGP41 產品與技術資訊](https://sensirion.com/products/catalog/SGP41)

## 9. 資料與研究聲明

本專題使用 MaleCNS v1.0 的神經元註解與腦圖譜資訊作為神經元候選篩選及視覺化依據。若再散布 MaleCNS 資料或其衍生內容，請依官方資料頁面所列授權與引用要求標示來源。

本專題自行建立的虛擬嗅覺受體、神經活動與 AI 回應結果，應清楚標示為模型輸出或預測結果；除非另有實驗證據，不應描述為真實果蠅的神經活動紀錄。
