#!/usr/bin/env python
# popmart_final_pipeline.py
# --------------------------------------------------
# Douyin  : Chinese BERT (fine-tuned + validation report)
# Reddit  : DistilBERT (fine-tuned + validation report)
# Visual  : automatic plt.show() for bar charts & wordclouds
# --------------------------------------------------

import os
import re
import emoji
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# ============== PATHS & CONSTS ==============
LABELED_CSV = "sample_for_manual_labeling.csv"
RAW_DOUYIN_CSV = "douyin_popmart_comments_all.csv"
RAW_REDDIT_CSV = "popmart_reddit_comments.csv"

BERT_CN_DIR = "sentiment_bert_chinese"
BERT_CN_NAME = "hfl/chinese-bert-wwm-ext"
BERT_EN_DIR = "sentiment_bert_english"
BERT_EN_NAME = "distilbert-base-uncased"
MAX_LEN = 128

DOUT_PRED_CSV = "douyin_with_bert_pred.csv"
REDD_PRED_CSV = "reddit_with_bert_pred.csv"
MERGED_CSV = "popmart_sentiment_final.csv"

label2id = {"positive": 0, "neutral": 1, "negative": 2}
id2label = {v: k for k, v in label2id.items()}

# ============== HELPERS ==============
url_pat = re.compile(r"http\S+")


def clean_cn(t: str) -> str:
    return emoji.replace_emoji(url_pat.sub("", str(t)), "")[:256]


def clean_en(t: str) -> str:
    txt = url_pat.sub("", str(t)).lower()
    txt = re.sub(r"[^a-z\s]", " ", txt)
    return " ".join(tok for tok in txt.split() if tok)


# 仅针对 Reddit 标签集
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

df = pd.read_csv(LABELED_CSV)
df = df[df.source == "reddit"].copy()
df["clean"] = df.comment.apply(clean_en)

X, y = df["clean"], df["label"]
pipe = make_pipeline(
    TfidfVectorizer(max_features=4000, ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000, class_weight="balanced")
)
# 过采样
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X.to_frame(), y)

pipe.fit(X_res["clean"], y_res)
print(classification_report(y, pipe.predict(X)))


# ============== CHINESE BERT ==============
def train_or_load_bert_cn():
    if os.path.isdir(BERT_CN_DIR):
        tok = AutoTokenizer.from_pretrained(BERT_CN_DIR)
        mdl = AutoModelForSequenceClassification.from_pretrained(BERT_CN_DIR)
        print("✓ Loaded cached Chinese BERT.")
    else:
        print("⏳ Fine-tuning Chinese BERT…")
        df = pd.read_csv(LABELED_CSV)
        df = df[df["source"] == "douyin"].copy()
        df["clean"] = df["comment"].apply(clean_cn)
        df["label_id"] = df["label"].map(label2id)
        tr, vl = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=42)

        tok = AutoTokenizer.from_pretrained(BERT_CN_NAME)
        mdl = AutoModelForSequenceClassification.from_pretrained(BERT_CN_NAME, num_labels=3)

        class DS(torch.utils.data.Dataset):
            def __init__(self, data):
                self.x = data["clean"].tolist()
                self.y = data["label_id"].tolist()

            def __len__(self): return len(self.x)

            def __getitem__(self, i):
                enc = tok(self.x[i], padding="max_length", truncation=True,
                          max_length=MAX_LEN, return_tensors="pt")
                return {
                    "input_ids": enc["input_ids"].squeeze(),
                    "attention_mask": enc["attention_mask"].squeeze(),
                    "labels": torch.tensor(self.y[i])
                }

        trainer = Trainer(
            model=mdl,
            args=TrainingArguments(
                output_dir="bert_cn_runs",
                num_train_epochs=5,
                per_device_train_batch_size=8,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                fp16=torch.cuda.is_available(),
                report_to="none"
            ),
            train_dataset=DS(tr),
            eval_dataset=DS(vl),
            data_collator=DataCollatorWithPadding(tokenizer=tok)
        )
        trainer.train()
        mdl.save_pretrained(BERT_CN_DIR)
        tok.save_pretrained(BERT_CN_DIR)
        print("✓ Chinese BERT fine-tuned and saved.")
    return AutoTokenizer.from_pretrained(BERT_CN_DIR), AutoModelForSequenceClassification.from_pretrained(BERT_CN_DIR)


# ============== ENGLISH DistilBERT ==============
def train_or_load_bert_en():
    if os.path.isdir(BERT_EN_DIR):
        tok = AutoTokenizer.from_pretrained(BERT_EN_DIR)
        mdl = AutoModelForSequenceClassification.from_pretrained(BERT_EN_DIR)
        print("✓ Loaded cached English DistilBERT.")
    else:
        print("⏳ Fine-tuning DistilBERT on Reddit…")
        df = pd.read_csv(LABELED_CSV)
        df = df[df["source"] == "reddit"].copy()
        df["clean"] = df["comment"].apply(clean_en)
        df["label_id"] = df["label"].map(label2id)
        tr, vl = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=42)

        tok = AutoTokenizer.from_pretrained(BERT_EN_NAME)
        mdl = AutoModelForSequenceClassification.from_pretrained(BERT_EN_NAME, num_labels=3)

        class RDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.texts = data["clean"].tolist()
                self.labels = data["label_id"].tolist()

            def __len__(self): return len(self.texts)

            def __getitem__(self, i):
                enc = tok(self.texts[i], padding="max_length", truncation=True,
                          max_length=MAX_LEN, return_tensors="pt")
                return {
                    "input_ids": enc["input_ids"].squeeze(),
                    "attention_mask": enc["attention_mask"].squeeze(),
                    "labels": torch.tensor(self.labels[i])
                }

        trainer = Trainer(
            model=mdl,
            args=TrainingArguments(
                output_dir="bert_en_runs",
                num_train_epochs=5,
                per_device_train_batch_size=8,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                fp16=torch.cuda.is_available(),
                report_to="none"
            ),
            train_dataset=RDataset(tr),
            eval_dataset=RDataset(vl),
            data_collator=DataCollatorWithPadding(tokenizer=tok)
        )
        trainer.train()
        mdl.save_pretrained(BERT_EN_DIR)
        tok.save_pretrained(BERT_EN_DIR)
        print("✓ English DistilBERT fine-tuned and saved.")
    return AutoTokenizer.from_pretrained(BERT_EN_DIR), AutoModelForSequenceClassification.from_pretrained(BERT_EN_DIR)


# ============== VALIDATION & PREDICTION ==============
def validate_and_predict_douyin(tok, mdl):
    df = pd.read_csv(LABELED_CSV)
    df = df[df["source"] == "douyin"].copy()
    df["clean"] = df["comment"].apply(clean_cn)
    df["label_id"] = df["label"].map(label2id)
    _, vl = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=42)

    mdl.eval()
    preds = []
    for i in range(0, len(vl), 32):
        texts = vl["clean"].iloc[i:i + 32].tolist()
        enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            logits = mdl(**enc).logits
        preds.extend(np.argmax(logits.cpu().numpy(), axis=1))
    print("\n— Douyin BERT validation —")
    print(classification_report(vl["label_id"], preds, target_names=label2id.keys()))
    plot_confusion(vl["label_id"], preds, title="Confusion Matrix – Douyin BERT Validation")

    dou = pd.read_csv(RAW_DOUYIN_CSV)
    dou["clean"] = dou["comment"].apply(clean_cn)
    preds = []
    for i in range(0, len(dou), 64):
        texts = dou["clean"].iloc[i:i + 64].tolist()
        enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            logits = mdl(**enc).logits
        preds.extend(np.argmax(logits.cpu().numpy(), axis=1))
    dou["bert_pred_id"] = preds
    dou["ml_pred"] = dou["bert_pred_id"].map(id2label)
    dou["source"] = "douyin"
    dou.to_csv(DOUT_PRED_CSV, index=False, encoding='utf-8-sig')
    print("✓ Douyin predictions saved →", DOUT_PRED_CSV)
    return dou


def validate_and_predict_reddit(tok, mdl):
    df = pd.read_csv(LABELED_CSV)
    df = df[df["source"] == "reddit"].copy()
    df["clean"] = df["comment"].apply(clean_en)
    df["label_id"] = df["label"].map(label2id)
    _, vl = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=42)

    mdl.eval()
    preds = []
    for i in range(0, len(vl), 32):
        texts = vl["clean"].iloc[i:i + 32].tolist()
        enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            logits = mdl(**enc).logits
        preds.extend(np.argmax(logits.cpu().numpy(), axis=1))
    print("\n— Reddit DistilBERT validation —")
    print(classification_report(vl["label_id"], preds, target_names=label2id.keys()))
    plot_confusion(vl["label_id"], preds, title="Confusion Matrix – Reddit BERT Validation")

    red = pd.read_csv(RAW_REDDIT_CSV)
    red["clean"] = red["comment"].apply(clean_en)
    preds = []
    for i in range(0, len(red), 64):
        texts = red["clean"].iloc[i:i + 64].tolist()
        enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            logits = mdl(**enc).logits
        preds.extend(np.argmax(logits.cpu().numpy(), axis=1))
    red["bert_pred_id"] = preds
    red["ml_pred"] = red["bert_pred_id"].map(id2label)
    red["source"] = "reddit"
    red.to_csv(REDD_PRED_CSV, index=False, encoding='utf-8-sig')
    print("✓ Reddit predictions saved →", REDD_PRED_CSV)
    return red


from sklearn.metrics import confusion_matrix


def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["positive", "neutral", "negative"],
                yticklabels=["positive", "neutral", "negative"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============== VISUALISATIONS ==============
def plot_bar(df, title):
    cnt = df["ml_pred"].value_counts().reindex(["positive", "neutral", "negative"]).fillna(0)
    sns.barplot(x=cnt.index, y=cnt.values)
    plt.title(title)
    plt.show()


def plot_wordcloud(df, sentiment, font_path=None):
    text = " ".join(df[df["ml_pred"] == sentiment]["comment"].astype(str))
    if not text: return
    wc = WordCloud(font_path=font_path, background_color="white").generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{sentiment} wordcloud")
    plt.show()


def plot_overall_wordcloud(df, title, font_path=None):
    """
    对整个 df['comment'] 生成一张词云，不按 sentiment 划分
    """
    text = " ".join(df["comment"].astype(str))
    if not text:
        return
    wc = WordCloud(font_path=font_path,
                   background_color="white").generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()


def run_topic_modeling(df, source, sentiment='all', num_topics=5, max_words=10):
    if sentiment != 'all':
        df = df[df["ml_pred"] == sentiment]

    # 使用英文或中文分词（下面以英文为例，中文要改为jieba分词）
    texts = df["comment"].dropna().tolist()

    # 简单的Bag-of-Words向量化
    vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=2)
    X = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    print(f"\n🧠 Topics for {source} - Sentiment = {sentiment}")
    words = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-max_words:][::-1]]
        print(f"  Topic {idx + 1}: {' '.join(top_words)}")


def extract_top_keywords(df, label, top_n=20):
    # 仅选择该情感类别
    texts = df[df["ml_pred"] == label]["comment"].astype(str).tolist()

    # 向量化
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)

    # 取均值排序
    tfidf_scores = tfidf_matrix.mean(axis=0).A1
    words = vectorizer.get_feature_names_out()
    sorted_indices = tfidf_scores.argsort()[::-1][:top_n]

    print(f"\n🔑 Top {top_n} keywords for sentiment = {label}")
    for i in sorted_indices:
        print(f"{words[i]} ({tfidf_scores[i]:.4f})")


# ============== RUN ==============
if __name__ == "__main__":
    # Chinese pipeline
    tok_cn, mdl_cn = train_or_load_bert_cn()
    dou_df = validate_and_predict_douyin(tok_cn, mdl_cn)

    # English pipeline
    tok_en, mdl_en = train_or_load_bert_en()
    red_df = validate_and_predict_reddit(tok_en, mdl_en)

    # Merge & save
    merged = pd.concat([dou_df[["comment", "source", "ml_pred"]],
                        red_df[["comment", "source", "ml_pred"]]],
                       ignore_index=True)
    merged.to_csv(MERGED_CSV, index=False, encoding='utf-8-sig')
    print("\n🎉 Merged data saved →", MERGED_CSV)

    # Optional Topic Modeling – Reddit Negative
    run_topic_modeling(red_df, source="Reddit", sentiment="positive")

    # Optional Topic Modeling – Douyin All
    run_topic_modeling(dou_df, source="Douyin", sentiment="positive")

    # 情绪分布柱状图
    plot_bar(dou_df, "Douyin Sentiment Distribution")
    plot_bar(red_df, "Reddit Sentiment Distribution")

    # 改为：两张整体词云
    plot_overall_wordcloud(dou_df, "Douyin Overall Word Cloud", font_path="simhei.ttf")
    plot_overall_wordcloud(red_df, "Reddit Overall Word Cloud")

    # 正负情绪关键词提取（用于报告撰写）
    extract_top_keywords(dou_df, "positive", top_n=15)
    extract_top_keywords(dou_df, "negative", top_n=15)

    extract_top_keywords(red_df, "positive", top_n=15)
    extract_top_keywords(red_df, "negative", top_n=15)

    print("\n✅ All done. Visualisations displayed.")
