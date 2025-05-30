# 🧠 Sentiment Analysis on Douyin & Reddit Comments (Pop Mart)

This repository contains code, data, and results for a sentiment analysis project comparing user comments from **Douyin** (Chinese platform) and **Reddit** (English forum), focused on the Pop Mart brand.

---

## 📁 File Overview

| File Name | Description |
|-----------|-------------|
| `sentiment_analysis.py` | Main pipeline script: model training, prediction, visualization |
| `user_comment_extraction.py` | Scraper used to extract comments from Douyin and Reddit |
| `sample_for_manual_labeling.csv` | Manually labeled dataset (used for BERT fine-tuning and validation) |
| `douyin_popmart_comments_all.csv` | Raw Douyin user comments |
| `popmart_reddit_comments.csv` | Raw Reddit user comments |
| `douyin_with_bert_pred.csv` | Douyin comments with BERT sentiment predictions |
| `reddit_with_bert_pred.csv` | Reddit comments with BERT sentiment predictions |
| `popmart_sentiment_final.csv` | Combined final output with all comments and predicted sentiments |

---

## 🔧 How to Use

1. Run `sentiment_analysis.py` to train the model or load saved ones.
2. Use the optional topic modeling and visualization functions included.
3. Ensure the `*.csv` files are in the same directory when running.

---

## 📚 Project Context

This project is part of a university assignment to:
- Compare lexicon-based and machine learning-based sentiment analysis
- Analyze multilingual, cross-platform emotional expression
- Identify emotional trends and key themes in user feedback

> All data is anonymized and used strictly for educational purposes.
