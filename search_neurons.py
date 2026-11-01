import pandas as pd
import numpy as np

# 讀取神經元註解資料
# file_path = "data/body-annotations-male-cns-v1.0-minconf-0.5.feather"

# neurons = pd.read_feather(file_path)

# 指定目前想研究的神經元分類
# target_classes = [
#     "ALIN",
#     "ALPN",
#     "MBON",
#     "DAN"
# ]

# 精確篩選 class
# olfactory_neurons = neurons[
#     neurons["class"].isin(target_classes)
# ].copy()

# 統計各類型數量
# print(
#     olfactory_neurons["class"].value_counts()
# )

# 儲存
# olfactory_neurons.to_csv(
#     "olfactory_neurons.csv",
#     index=False,
#     encoding="utf-8-sig"
# )

# 和Github的template比對

# ==================================
# 1. 讀取剛剛產生的神經元 CSV
# ==================================

df = pd.read_csv("olfactory_neurons.csv")

# 確保 bodyId 是整數
df["bodyId"] = pd.to_numeric(
    df["bodyId"],
    errors="raise"
).astype("int64")

# ==================================
# 2. 讀取 GitHub 的神經元資料
# ==================================

root = "fly-connectome-template/public/data/brain-atlas/"

ids = np.fromfile(
    root + "ids.bin",
    dtype="<u4"
)

groups = np.fromfile(
    root + "groups.bin",
    dtype="u1"
)

# 確認檔案數量相同，相同 = 成功讀取
# print("神經元 ID 數量：", len(ids))
# print("群組資料數量：", len(groups))

# print("前 10 個神經元 ID：", ids[:10])
# print("前 10 個群組：", groups[:10])

print("候選神經元總數：", len(df))

# ==================================
# 3. 找出 GitHub 可以顯示的神經元
# ==================================

visible_ids = ids[groups < 3]

print("GitHub 可顯示的神經元數量：",
      len(visible_ids))


# ==================================
# 4. 比對 bodyId
# ==================================

df["is_visible"] = df["bodyId"].isin(
    visible_ids
)


# ==================================
# 5. 統計各種類別的比對結果
# ==================================

summary = df.groupby("class")["is_visible"].agg(
    total="count",
    visible="sum"
)

summary["not_visible"] = (
    summary["total"] - summary["visible"]
)
print("\n各類神經元比對結果：")
print(summary)


# ==================================
# 6. 儲存可以顯示的神經元
# ==================================

visible_olfactory = df[
    df["is_visible"]
].copy()

visible_olfactory.to_csv(
    "visible_olfactory_neurons.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n成功產生 visible_olfactory_neurons.csv")
print("可顯示的候選神經元：", len(visible_olfactory))

# 建立 GitHub 神經元 ID 與群組對照表
atlas = pd.DataFrame({
    "bodyId": ids.astype("int64"),
    "template_group": groups.astype("int64")
})

# 將候選神經元與 GitHub 資料合併
check = df.merge(
    atlas,
    on="bodyId",
    how="left",
    indicator=True
)

# 判斷是否存在於 GitHub
check["in_template"] = check["_merge"] == "both"

# 判斷是否符合顯示條件
check["passes_display_rule"] = (
    check["in_template"] &
    (check["template_group"] < 3)
)

# 找出不能顯示的神經元
excluded = check[
    ~check["passes_display_rule"]
].copy()

# 判斷不能顯示的原因
excluded["reason"] = np.where(
    ~excluded["in_template"],
    "ID 不存在於 ids.bin",
    "groups >= 3"
)

print("\n不能顯示的神經元：")

print(
    excluded[
        [
            "bodyId",
            "type",
            "class",
            "template_group",
            "reason"
        ]
    ].to_string(index=False)
)