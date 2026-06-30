#count = 0
#total = 0

#with open("phishguard_gh_dataset.csv", "r", encoding="utf-8") as f:
#    for line in f:
#        total += 1
#        if "http://" in line.lower() or "https://" in line.lower():
#            count += 1

#print("Total rows:", total)
#print("Rows with scheme:", count)
#print("Percentage:", round((count/total)*100, 2), "%")*/

# import csv

# phish_scheme = 0
# legit_scheme = 0

# with open("phishguard_gh_dataset.csv", encoding="utf-8") as f:
#     reader = csv.reader(f)

#     for row in reader:
#         text = ",".join(row).lower()

#         if "http://" in text or "https://" in text:
#             if "phish" in text or "1" in row:
#                 phish_scheme += 1
#             else:
#                 legit_scheme += 1

# print("Phishing with scheme:", phish_scheme)
# print("Legitimate with scheme:", legit_scheme)

# import pandas as pd

# df = pd.read_csv("phishguard_gh_dataset.csv")

# # adjust column name if needed
# url_col = "url"
# label_col = "label"

# df["length"] = df[url_col].astype(str).apply(len)

# print("Average URL length:")
# print(df.groupby(label_col)["length"].mean())

# print("\nLongest URLs:")
# print(
#     df.sort_values("length", ascending=False)
#     [[url_col, label_col, "length"]]
#     .head(20)
# )


# import pandas as pd

# df = pd.read_csv("phishguard_gh_dataset.csv")

# df["symbols"] = df["url"].astype(str).apply(
#     lambda x: sum(1 for c in x if c in "@?&=_-%")
# )

# print(df.groupby("label")["symbols"].mean())

# import pandas as pd

# df = pd.read_csv("phishguard_gh_dataset.csv")

# keywords = [
#     "login",
#     "signin",
#     "verify",
#     "account",
#     "secure",
#     "update",
#     "password",
#     "paypal",
#     "bank",
#     "confirm"
# ]

# for word in keywords:
#     df[word] = df["url"].astype(str).str.lower().str.contains(word)

# print("Keyword phishing rates\n")

# for word in keywords:
#     result = df.groupby(word)["label"].mean()
#     print(word)
#     print(result)
#     print()

# 

import pandas as pd
from urllib.parse import urlparse

df = pd.read_csv("phishguard_gh_dataset.csv")


def get_tld(x):
    try:
        x = str(x)
        if "://" not in x:
            x = "http://" + x

        domain = urlparse(x).netloc

        return domain.split(".")[-1]

    except:
        return "invalid"


df["tld"] = df["url"].apply(get_tld)


print(
    df.groupby("tld")["label"]
    .agg(["count","mean"])
    .sort_values("count", ascending=False)
    .head(20)
)