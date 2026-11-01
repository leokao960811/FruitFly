import pandas as pd
import numpy as np

# 讀取 MaleCNS 註解資料
file_path = (
    "data/body-annotations-male-cns-v1.0-minconf-0.5.feather"
)

neurons = pd.read_feather(file_path)

# 檢查有哪些欄位
print("資料欄位：")
print(neurons.columns.tolist())

# 搜尋可能出現 Kenyon 的欄位
search_cols = [
    "class",
    "subclass",
    "superclass",
    "type",
    "instance",
    "hemibrainType",
    "flywireType",
    "synonyms"
]

for col in search_cols:

    if col not in neurons.columns:
        continue

    matches = (
        neurons[col]
        .dropna()
        .astype(str)
    )

    matches = matches[
        matches.str.contains(
            "Kenyon",
            case=False,
            na=False
        )
    ].unique()

    if len(matches) > 0:
        print(f"\n欄位：{col}")
        print(matches[:30])

# ==================================
# 搜尋 Kenyon Cells
# ==================================

# 統一格式，避免大小寫、空格或底線造成差異
class_normalized = (
    neurons["class"]
    .astype("string")
    .str.replace(r"[\s_-]+", "", regex=True)
    .str.casefold()
)

# 精確篩選 Kenyon Cell
kc_neurons = neurons[
    class_normalized == "kenyoncell"
].copy()

# print("\nKenyon Cells 搜尋結果")
# print("KC 總數：", len(kc_neurons))

# 檢查各種 KC type 的數量
if "type" in kc_neurons.columns:

    print("\nKC 類型統計：")
    print(kc_neurons["type"].value_counts())

# 顯示前 20 筆資料
show_cols = [
    "bodyId",
    "type",
    "instance",
    "superclass",
    "class",
    "hemibrainType",
    "flywireType",
    "somaSide",
    "status"
]

show_cols = [
    col for col in show_cols
    if col in kc_neurons.columns
]

# print("\n前 20 筆 KC：")

# print(
#     kc_neurons[show_cols]
#     .head(20)
#     .to_string(index=False)
# )

# 儲存 KC 資料
# kc_neurons.to_csv(
#     "kc_neurons.csv",
#     index=False,
#     encoding="utf-8-sig"
# )

# print("\n成功產生 kc_neurons.csv")

# =================================
# Step 1. 讀取 KC 候選清單
# =================================

kc = pd.read_csv("kc_neurons.csv")

kc["bodyId"] = pd.to_numeric(
    kc["bodyId"],
    errors="raise"
).astype("int64")

print("KC 候選神經元總數：", len(kc))


# =================================
# Step 2. 讀取 GitHub 腦圖譜
# =================================

root = "fly-connectome-template/public/data/brain-atlas/"

ids = np.fromfile(
    root + "ids.bin",
    dtype="<u4"
)

groups = np.fromfile(
    root + "groups.bin",
    dtype="u1"
)

assert len(ids) == len(groups)

# GitHub 中所有神經元 ID
all_ids = set(
    ids.astype("int64")
)

# 符合 GitHub 顯示條件的神經元
visible_ids = set(
    ids[groups < 3].astype("int64")
)


# =================================
# Step 3. 比對 KC 的 bodyId
# =================================

kc["in_template"] = (
    kc["bodyId"].isin(all_ids)
)

kc["is_visible"] = (
    kc["bodyId"].isin(visible_ids)
)


# =================================
# Step 4. 統計結果
# =================================

summary = kc.groupby("type")["is_visible"].agg(
    total="count",
    visible="sum"
)

summary["not_visible"] = (
    summary["total"] - summary["visible"]
)

print("\nKC 各類型比對結果：")
print(summary)

print("\nKC 總數：", len(kc))
print("GitHub 可顯示的 KC：",
      kc["is_visible"].sum())


# =================================
# Step 5. 儲存結果
# =================================

visible_kc = kc[
    kc["is_visible"]
].copy()

excluded_kc = kc[
    ~kc["is_visible"]
].copy()

visible_kc.to_csv(
    "visible_kc_neurons.csv",
    index=False,
    encoding="utf-8-sig"
)

excluded_kc.to_csv(
    "excluded_kc_neurons.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n成功產生：")
print("visible_kc_neurons.csv")
print("excluded_kc_neurons.csv")

# 排除原因

# 建立 GitHub ID 與群組對照表
atlas = pd.DataFrame({
    "bodyId": ids.astype("int64"),
    "template_group": groups.astype("int64")
})

# 避免重複 ID 造成合併問題
atlas = atlas.drop_duplicates(
    subset=["bodyId"]
)

# 比對 KC 與 GitHub
kc_check = kc.merge(
    atlas,
    on="bodyId",
    how="left",
    indicator=True
)

# 判斷未顯示原因
kc_check["reason"] = "可顯示"

kc_check.loc[
    kc_check["_merge"] == "left_only",
    "reason"
] = "ID 不存在於 ids.bin"

kc_check.loc[
    (kc_check["_merge"] == "both") &
    (kc_check["template_group"] >= 3),
    "reason"
] = "groups >= 3"

# 取出未顯示的 KC
excluded_kc = kc_check[
    kc_check["reason"] != "可顯示"
].copy()

# 顯示結果
print("未顯示 KC 數量：", len(excluded_kc))

print(
    excluded_kc[
        [
            "bodyId",
            "type",
            "template_group",
            "reason"
        ]
    ].to_string(index=False)
)

# 統計排除原因
print("\n排除原因統計：")
print(excluded_kc["reason"].value_counts())

# 儲存診斷結果
excluded_kc.to_csv(
    "excluded_kc_diagnosis.csv",
    index=False,
    encoding="utf-8-sig"
)