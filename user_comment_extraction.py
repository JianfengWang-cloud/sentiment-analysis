import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import re
import pandas as pd  # ✅ 新增模块

nltk.download('vader_lexicon')

# Reddit 认证信息
reddit = praw.Reddit(
    client_id="y5XKXy1o-wuF7bazjowTmQ",
    client_secret="FnpaoYnt01jMT5jJORkitNU9ow--UQ",
    username="PruneOk9023",
    password="Aa!270032",
    user_agent="popmart-nlp/0.1 by PruneOk9023"
)

# 目标帖子链接
url = "https://www.reddit.com/r/PopMartCollectors/comments/1hoh09e/im_new_to_pop_mart_collecting_and_i_was_sold_a/"
submission = reddit.submission(url=url)

# 展开所有评论
submission.comments.replace_more(limit=None)
comments = [comment.body for comment in submission.comments]

print(f"Total comments scraped: {len(comments)}")

# ✅ 保存原始评论到 CSV 文件
df = pd.DataFrame({'comment': comments})
df.to_csv("popmart_reddit_comments5.csv", index=False, encoding='utf-8-sig')  # 保存为 UTF-8 带 BOM 以兼容 Excel
print("Comments saved to popmart_reddit_comments.csv")

# 文本清洗函数
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # 移除链接
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # 移除标点和非字母字符
    return text.lower()

cleaned_comments = [clean_text(c) for c in comments]

# 情感分析
sia = SentimentIntensityAnalyzer()
scores = [sia.polarity_scores(c)['compound'] for c in cleaned_comments]

# 统计正面/中性/负面比例
positive = sum(1 for s in scores if s > 0.05)
neutral = sum(1 for s in scores if -0.05 <= s <= 0.05)
negative = sum(1 for s in scores if s < -0.05)

print(f"Positive: {positive} | Neutral: {neutral} | Negative: {negative}")

# 生成词云
text_blob = " ".join(cleaned_comments)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_blob)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud of Pop Mart Reddit Comments", fontsize=16)
plt.show()
