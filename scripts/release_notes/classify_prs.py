# In[1]:
import pandas as pd

# In[2]:
data_filename = "data.json"
df = pd.read_json(data_filename).T
df.tail()

# In[3]:
all_labels = {lbl for labels in df["labels"] for lbl in labels}
all_labels

# In[4]:
# Add one column per label
for label in all_labels:
    df[label] = df["labels"].apply(lambda labels_list: label in labels_list)
df.head()

# In[5]:
# Add a clean "module" column. It contains tuples since PRs can have more than one module.
# Maybe we should include "topics" in that column as well?

all_modules = {  # mapping: full name -> clean name
    label: "".join(label.split(" ")[1:]) for label in all_labels if label.startswith("module")
}

# We use an ugly loop, but whatever ¯\_(ツ)_/¯
df["module"] = [[] for _ in range(len(df))]
for i, row in df.iterrows():
    for full_name, clean_name in all_modules.items():
        if full_name in row["labels"]:
            row["module"].append(clean_name)
df["module"] = df.module.apply(tuple)
df.head()

# In[6]:
mod_df = df.set_index("module").sort_index()
mod_df.tail()

# In[7]:
# All improvement PRs
mod_df[mod_df["enhancement"]].head()

# In[8]:
# improvement f module
# note: don't filter module name on the index as the index contain tuples with non-exclusive values
# Use the boolean column instead
mod_df[mod_df["enhancement"] & mod_df["module: transforms"]]


# In[9]:
def format_prs(mod_df, exclude_prototype=True):
    out = []
    for idx, row in mod_df.iterrows():
        if exclude_prototype and "prototype" in row and row["prototype"]:
            continue
        modules = idx
        # Put "documentation" and "tests" first for sorting to be dece
        for last_module in ("documentation", "tests"):
            if last_module in modules:
                modules = [m for m in modules if m != last_module] + [last_module]

        module = f"[{', '.join(modules)}]"
        module = module.replace("referencescripts", "reference scripts")
        module = module.replace("code", "reference scripts")
        out.append(f"{module} {row['title']}")

    return "\n".join(out)


# In[10]:
included_prs = pd.DataFrame()

# If labels are accurate, this shouhld generate most of the release notes already
# We keep track of the included PRs to figure out which ones are missing
for section_title, module_idx in (
    ("Backward-incompatible changes", "bc-breaking"),
    ("Deprecations", "deprecation"),
    ("New Features", "new feature"),
    ("Improvements", "enhancement"),
    ("Bug Fixes", "bug"),
    ("Code Quality", "code quality"),
):
    if module_idx in mod_df:
        print(f"## {section_title}")
        print()
        tmp_df = mod_df[mod_df[module_idx]]
        included_prs = pd.concat([included_prs, tmp_df])
        print(format_prs(tmp_df))
        print()


# In[11]:
# Missing PRs are these ones... classify them manually
missing_prs = pd.concat([mod_df, included_prs]).drop_duplicates(subset="pr_number", keep=False)
print(format_prs(missing_prs))

# In[12]:
# Generate list of contributors
print()
print("## Contributors")

previous_release = "c35d3855ccbfa6a36e6ae6337a1f2c721c1f1e78"
current_release = "5181a854d8b127cf465cd22a67c1b5aaf6ccae05"
print(
    f"{{ git shortlog -s {previous_release}..{current_release} | cut -f2- & git log -s {previous_release}..{current_release} | grep Co-authored | cut -f2- -d: | cut -f1 -d\\< | sed 's/^ *//;s/ *//' ; }} | sort --ignore-case | uniq | tr '\\n' ';' | sed 's/;/, /g;s/,//' | fold -s"
)

# In[13]:
# Utility to extract PR numbers only from multiple lines, useful to bundle all
# the docs changes for example:
import re

s = """

[] Remove unnecessary dependency from macOS/Conda binaries (#8077)
[rocm] [ROCm] remove HCC references (#8070)
"""

print(", ".join(re.findall("(#\\d+)", s)))
