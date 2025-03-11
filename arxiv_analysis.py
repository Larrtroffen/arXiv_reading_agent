# arxiv_api_analysis.py
import os
import time
import xmltodict
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from datetime import datetime, timedelta
from typing import List, Dict
from gradio_client import Client
from openai import OpenAI
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from tqdm import tqdm
import requests
import pandas as pd

matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class ArxivAPI:
    """arXiv API客户端"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.max_retries = 3
        self.delay = 3

    def fetch_recent_articles(self, category: str = "cs.CY") -> List[Dict]:
        """获取过去两周文章（UTC时间）"""
        now_utc = datetime.now().astimezone()
        today_utc = now_utc.date()
        start_date_utc = today_utc - timedelta(days=13)

        date_format = "%Y%m%d"
        query = f"cat:{category} AND submittedDate:[{start_date_utc.strftime(date_format)} TO {today_utc.strftime(date_format)}]"
        
        params = {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": 0,
            "max_results": 200
        }
        articles = []

        while True:
            print(f"\n[API请求] 获取 {params['start']+1}-{params['start']+params['max_results']} 条数据...")
            response = self._request_with_retry(params)
            
            if not response or response.status_code != 200:
                break

            data = xmltodict.parse(response.text)
            if 'feed' not in data or 'entry' not in data['feed']:
                break

            entries = data['feed']['entry']
            entries = [entries] if isinstance(entries, dict) else entries

            if not entries:
                break

            for entry in entries:
                try:
                    pub_date = datetime.strptime(
                        entry['published'], 
                        '%Y-%m-%dT%H:%M:%SZ'
                    ).date()
                    
                    authors = []
                    if 'author' in entry:
                        author_data = entry['author']
                        if isinstance(author_data, dict):
                            authors.append(author_data.get('name', 'Unknown'))
                        elif isinstance(author_data, list):
                            authors = [a.get('name', 'Unknown') for a in author_data]
                    
                    article = {
                        'arxiv_id': entry['id'].split('/')[-1],
                        'title': entry['title'].strip(),
                        'authors': authors,
                        'abstract': entry['summary'].strip(),
                        'date': pub_date.strftime("%Y-%m-%d"),
                        'published': entry['published']
                    }
                    articles.append(article)
                except Exception as e:
                    print(f"[解析错误] {str(e)}")

            if len(entries) < params['max_results']:
                break

            params["start"] += params["max_results"]
            time.sleep(self.delay)

        print(f"\n[API汇总] 共获取 {len(articles)} 篇过去两周的文章")
        return articles

    def _request_with_retry(self, params: Dict) -> requests.Response:
        for i in range(self.max_retries):
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                return response
            except Exception as e:
                print(f"\n[API错误] 第{i+1}次尝试失败: {str(e)}")
                if i < self.max_retries - 1:
                    time.sleep(2 ** i)
        return None

class ArxivAnalyzer:
    """arXiv文章分析系统"""
    
    def __init__(self):
        self.articles = []
        self.translator = OpenAI(
            base_url="https://api.openai-proxy.org/v1",
            api_key='sk-lhmsRn3GdhRC77VaDK6XgS29aMn5fsxbXy6OnYA6JUGw2fii',
            timeout=30
        )
        self.eval_client = Client("ssocean/Newborn_Article_Impact_Predict")
        self._configure_chinese_font()

    def _configure_chinese_font(self):
        """配置中文字体"""
        try:
            if os.name == 'nt':
                font_path = 'C:/Windows/Fonts/simhei.ttf'
            else:
                font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
            
            self.font_prop = FontProperties(fname=font_path)
            plt.rcParams['font.sans-serif'] = [self.font_prop.get_name()]
            self.wordcloud_font = font_path
            print(f"[系统] 使用字体：{self.font_prop.get_name()}")
        except Exception as e:
            print(f"[警告] 字体加载失败：{str(e)}")
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
            self.font_prop = FontProperties()
            self.wordcloud_font = None

    def run_pipeline(self):
        """完整处理流程"""
        print("\n=== 阶段1: 数据获取 ===")
        api = ArxivAPI()
        self.articles = api.fetch_recent_articles()
        if not self.articles:
            print("\n[警告] 过去两周无新文章，流程终止")
            return

        print("\n=== 阶段2: 数据统计 ===")
        self.analyze_daily_distribution()

        print("\n=== 阶段3: 中英翻译 ===")
        self.translate_articles()

        print("\n=== 阶段4: 主题聚类 ===")
        self.analyze_topics()

        print("\n=== 阶段5: 质量评估 ===")
        top_articles = self.evaluate_articles()

        print("\n=== 阶段6: 生成报告 ===")
        self.generate_report(top_articles)
        
        print("\n=== 阶段7: 数据保存 ===")
        self.save_to_excel()

    def analyze_daily_distribution(self):
        """统计每日文章数量"""
        dates = sorted({a['date'] for a in self.articles})
        if not dates:
            return
            
        start_date = min(datetime.strptime(d, "%Y-%m-%d") for d in dates)
        date_range = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(14)]
        
        daily_counts = {d:0 for d in date_range}
        for article in self.articles:
            daily_counts[article['date']] += 1
        
        sorted_dates = sorted(daily_counts.keys(), reverse=True)
        counts = [daily_counts[d] for d in sorted_dates]
        
        self._plot_daily_distribution(sorted_dates, counts)

    def _plot_daily_distribution(self, dates: List[str], counts: List[int]):
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(dates)), counts, color='#4C72B0')
        
        plt.title('过去两周每日文章数量', fontproperties=self.font_prop, fontsize=14)
        plt.xlabel('日期', fontproperties=self.font_prop)
        plt.ylabel('文章数量', fontproperties=self.font_prop)
        plt.xticks(range(len(dates)), [d.split('-')[-1] for d in dates], rotation=45)
        
        for i, v in enumerate(counts):
            plt.text(i, v+0.5, str(v), ha='center', fontsize=9)
            
        plt.tight_layout()
        plt.savefig('daily_distribution.png', dpi=120)
        plt.close()

    def translate_articles(self):
        progress = tqdm(self.articles, desc="翻译进度", unit="篇")
        for article in progress:
            try:
                article['title_zh'] = self._translate_text(article['title'], is_title=True)
                article['abstract_zh'] = self._translate_text(article['abstract'])
                time.sleep(1)
            except Exception as e:
                print(f"\n[翻译错误] {str(e)}")

    def _translate_text(self, text: str, is_title: bool = False) -> str:
        prompt = "专业学术翻译，保留英文术语，" + ("标题用（）包含英文" if is_title else "摘要完整翻译")
        response = self.translator.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content

    def analyze_topics(self):
        texts = [f"{a['title']} {a['abstract']}" for a in self.articles]
        
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        X = vectorizer.fit_transform(texts)

        n_clusters = min(5, len(self.articles))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # 添加聚类标签到文章数据
        for i, article in enumerate(self.articles):
            article['cluster'] = int(clusters[i])
            
        # 提取聚类关键词
        feature_names = vectorizer.get_feature_names_out()
        self.cluster_keywords = {}
        
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            if not cluster_indices:
                continue
                
            cluster_X = X[cluster_indices]
            avg_tfidf = cluster_X.mean(axis=0).A1
            sorted_indices = avg_tfidf.argsort()[::-1]
            top_words = [(feature_names[i], avg_tfidf[i]) for i in sorted_indices[:10]]
            self.cluster_keywords[cluster_id] = top_words

        self._generate_wordcloud(texts)
        self._plot_clusters(clusters)
        self._plot_cluster_keywords()  # 新增关键词可视化

    def _plot_cluster_keywords(self):
        """绘制每个聚类的关键词"""
        plt.figure(figsize=(12, 6 * len(self.cluster_keywords)))
        
        for idx, (cluster_id, words_scores) in enumerate(self.cluster_keywords.items(), 1):
            plt.subplot(len(self.cluster_keywords), 1, idx)
            words = [ws[0] for ws in words_scores]
            scores = [ws[1] for ws in words_scores]
            
            plt.barh(range(len(words)), scores, color='#4C72B0')
            plt.yticks(range(len(words)), words, fontproperties=self.font_prop)
            plt.gca().invert_yaxis()
            plt.title(f'主题 {cluster_id} 关键词分布', fontproperties=self.font_prop)
            plt.xlabel('TF-IDF 重要性', fontproperties=self.font_prop)
        
        plt.tight_layout()
        plt.savefig('cluster_keywords.png', dpi=120)
        plt.close()

    def save_to_excel(self):
        """保存数据到Excel文件"""
        
        # 文章数据
        articles_data = [{
            'arXiv ID': a['arxiv_id'],
            '英文标题': a['title'],
            '中文标题': a.get('title_zh', ''),
            '作者': ', '.join(a['authors']),
            '发布日期': a['date'],
            '评分': a.get('score', 0),
            '所属主题': a.get('cluster', -1),
            '英文摘要': a['abstract'],
            '中文摘要': a.get('abstract_zh', '')
        } for a in self.articles]

        # 主题关键词数据
        topics_data = [{
            '主题ID': cluster_id,
            '关键词': ', '.join([f"{word}({score:.2f})" for word, score in words_scores]),
            '文章数量': sum(1 for a in self.articles if a.get('cluster') == cluster_id)
        } for cluster_id, words_scores in self.cluster_keywords.items()]

        # 每日统计
        daily_stats = {}
        for a in self.articles:
            daily_stats[a['date']] = daily_stats.get(a['date'], 0) + 1
        daily_data = [{'日期': k, '文章数量': v} for k, v in daily_stats.items()]

        with pd.ExcelWriter('arxiv_analysis.xlsx') as writer:
            pd.DataFrame(articles_data).to_excel(writer, sheet_name='文章数据', index=False)
            pd.DataFrame(topics_data).to_excel(writer, sheet_name='主题分析', index=False)
            pd.DataFrame(daily_data).to_excel(writer, sheet_name='每日统计', index=False)

    def _generate_wordcloud(self, texts: List[str]):
        wordcloud = WordCloud(
            font_path=self.wordcloud_font,
            width=1200,
            height=600,
            background_color='white',
            collocations=False
        ).generate(' '.join(texts))
        
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.savefig('wordcloud.png', bbox_inches='tight')
        plt.close()

    def _plot_clusters(self, clusters: List[int]):
        cluster_counts = {i: list(clusters).count(i) for i in set(clusters)}
        
        plt.figure(figsize=(10, 6))
        plt.bar(cluster_counts.keys(), cluster_counts.values())
        plt.title('主题聚类分布', fontproperties=self.font_prop)
        plt.xlabel('聚类编号', fontproperties=self.font_prop)
        plt.ylabel('文章数量', fontproperties=self.font_prop)
        plt.savefig('cluster_dist.png', bbox_inches='tight')
        plt.close()

    def evaluate_articles(self) -> List[Dict]:
        top_articles = []
        progress = tqdm(self.articles, desc="评估进度", unit="篇")
        
        for article in progress:
            try:
                result = self.eval_client.predict(
                    title=article['title'],
                    abstract=article['abstract'],
                    api_name="/predict"
                )
                article['score'] = float(result['label'])
            except Exception as e:
                article['score'] = 0.0
                
            time.sleep(0.5)

        return sorted(self.articles, key=lambda x: x['score'], reverse=True)[:10]

    def generate_report(self, top_articles: List[Dict]):
        date_str = datetime.now().strftime("%Y年%m月%d日")
        
        md_content = f"""# 过去两周计算机与社会 arXiv 文章
（{date_str}）
**文章总数**: {len(self.articles)}篇  
**推荐阈值**: {top_articles[-1]['score']:.2f}分以上  

## 发文趋势分析
![每日发文量](daily_distribution.png)

## 主题分析
![主题分布](cluster_dist.png)
![主题关键词](cluster_keywords.png)"""

# 精选推荐"""
        
        for idx, article in enumerate(top_articles, 1):
            md_content += f"""
### 推荐论文 {idx}: {article['title_zh']}
**评分**: {article['score']:.2f}  
**作者**: {', '.join(article['authors']) if article['authors'] else '未知作者'}  

**摘要**:  
{article['abstract_zh']}

---
"""

        with open('daily_report.md', 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        print(f"\n[报告] 已生成 daily_report.md")

if __name__ == "__main__":
    analyzer = ArxivAnalyzer()
    analyzer.run_pipeline()