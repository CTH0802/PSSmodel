import re

# 路徑修改成你的檔案
tex_file = "Thesis.tex"
bib_file = "reference.bib"

# 1. 讀取 tex 檔，找出所有 \cite, \citep, \citet 等引用的 key
with open(tex_file, encoding="utf-8") as f:
    tex_content = f.read()

# 正則抓取所有 \cite 形式的引用
cite_keys = re.findall(r'\\cite[t|p|alp]*\{([^}]*)\}', tex_content)
# 可能同一個 {} 內有多個 key，用 , 分隔
cite_keys_split = []
for k in cite_keys:
    cite_keys_split.extend([x.strip() for x in k.split(",")])

cite_keys_split = set(cite_keys_split)
print(f"Tex 中引用的 key 共 {len(cite_keys_split)} 個:")
print(cite_keys_split)

# 2. 讀取 bib 檔，抓出所有 entry 的 key
with open(bib_file, encoding="utf-8") as f:
    bib_content = f.read()

bib_keys = re.findall(r'@\w+\{([^,]+),', bib_content)
bib_keys = set(bib_keys)
print(f"Bib 檔中總共有 {len(bib_keys)} 個 key:")
print(bib_keys)

# 3. 找出 tex 裡有但 bib 裡沒有的 key
missing_keys = cite_keys_split - bib_keys
if missing_keys:
    print("\n警告！以下 citation key 在 bib 檔找不到：")
    for k in missing_keys:
        print(k)
else:
    print("\n所有引用 key 都在 bib 檔中找到 ✅")
