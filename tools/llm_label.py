import json

import pandas as pd
from openai import OpenAI

client = OpenAI(api_key="YOUR_KEY")


def call_gpt(prompt):
    # 调用OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 使用GPT-4 Turbo模型
        messages=[{"role": "user", "content": prompt}],
    )

    # 获取模型的响应
    model_response = response.choices[0].message.content
    print("-------------------------------")
    # print(prompt)
    # print('--------------')
    print(model_response)
    print("-------------------------------")

    return model_response


def post_process(text):
    if ":" in text:
        text = text.split(":", 1)[1]
    if "is" in text:
        text = text.split("is", 1)[1]
    if "are" in text:
        text = text.split("are", 1)[1]
    labels = text.split(",")
    labels = [l.strip().lower() for l in labels]
    return labels


metadata = pd.read_csv("data/all_dataset_description.csv", index_col=0)
metadata.set_index(keys="did", inplace=True)

with open("data/merged_labels.json", "r") as f:
    merged_labels = json.load(f)

TEMPLATE = "Given the description of a dataset, select relevant labels from the candidates. Directly output the labels, separated by commas.\nDescription: [[DESC]]\nCandidates: [[CANDIDATES]]"

# Get all category names
topic_path = "data/LLM Evaluation - Topic Queries.csv"
df = pd.read_csv(topic_path)
CLS = df["Topic"].unique().tolist()
lower_CLS = [c.strip().lower() for c in CLS]
candidates = ", ".join(CLS)
llm_labels = {}
for did in merged_labels:
    print(did)
    desc = metadata.loc[int(did)]["description"]
    prompt = TEMPLATE.replace("[[DESC]]", desc).replace("[[CANDIDATES]]", candidates)
    response = call_gpt(prompt)
    response = post_process(response)
    labels = []
    for l in response:
        if l in lower_CLS:
            labels.append(CLS[lower_CLS.index(l)])
    if len(labels) > 0:
        llm_labels[did] = labels

with open("data/llm_labels.json", "w") as f:
    json.dump(llm_labels, f)
