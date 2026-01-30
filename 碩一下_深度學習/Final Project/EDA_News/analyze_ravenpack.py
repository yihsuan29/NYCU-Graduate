import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# 載入 Parquet 檔案
def load_and_analyze_ravenpack():
    print("開始載入 RavenPack 資料...")
    try:
        ravenpack_data = pd.read_parquet('DataExample/RavenPack/2000.pqt')
        print(f"成功載入資料：{ravenpack_data.shape[0]:,} 筆記錄，{ravenpack_data.shape[1]} 個欄位")
        return ravenpack_data
    except Exception as e:
        print(f"載入資料時發生錯誤: {e}")
        return None

# 基本資料探索
def explore_basic_info(df):
    print("\n" + "="*50)
    print("基本資料資訊")
    print("="*50)
    
    # 檢查資料型態
    print("\n資料型態：")
    print(df.dtypes)
    
    # 檢查缺失值
    print("\n缺失值統計：")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_info = pd.DataFrame({
        '缺失值數量': missing_values,
        '缺失百分比': missing_percent.round(2)
    })
    print(missing_info[missing_info['缺失值數量'] > 0].sort_values('缺失百分比', ascending=False))
    
    # 基本統計資訊 (針對數值型欄位)
    print("\n數值型欄位統計資訊：")
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    print(df[numeric_columns].describe().transpose())
    
    # 檢視前五筆資料
    print("\n前五筆資料：")
    print(df.head())

# 時間分析
def analyze_time_distribution(df):
    print("\n" + "="*50)
    print("時間分佈分析")
    print("="*50)
    
    # 確保日期欄位格式正確
    df['DATE'] = pd.to_datetime(df['RPA_DATE_UTC'])
    df['TIME'] = pd.to_datetime(df['RPA_TIME_UTC'], format='%H:%M:%S.%f').dt.time
    
    # 按日期分組
    daily_counts = df.groupby('DATE').size()
    
    print(f"\n日期範圍：{df['DATE'].min()} 至 {df['DATE'].max()}")
    print(f"總天數：{len(daily_counts)} 天")
    print(f"每日平均新聞數量：{daily_counts.mean():.2f}")
    print(f"每日新聞數量中位數：{daily_counts.median()}")
    print(f"每日最多新聞數量：{daily_counts.max()} (日期：{daily_counts.idxmax()})")
    
    # 每日新聞趨勢圖
    plt.figure(figsize=(15, 6))
    daily_counts.plot()
    plt.title('每日新聞數量分佈')
    plt.xlabel('日期')
    plt.ylabel('新聞數量')
    plt.tight_layout()
    plt.savefig('ravenpack_daily_news_count.png')
    
    # 星期幾分佈
    df['WEEKDAY'] = df['DATE'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = df['WEEKDAY'].value_counts().reindex(weekday_order)
    
    print("\n星期幾分佈：")
    for day, count in weekday_counts.items():
        print(f"{day}: {count:,} 筆 ({count/len(df)*100:.2f}%)")
    
    # 時間分佈 (小時)
    df['HOUR'] = pd.to_datetime(df['RPA_TIME_UTC'], format='%H:%M:%S.%f').dt.hour
    hourly_counts = df['HOUR'].value_counts().sort_index()
    
    print("\n每小時新聞分佈：")
    for hour, count in hourly_counts.items():
        print(f"{hour:02d}:00-{hour+1:02d}:00: {count:,} 筆 ({count/len(df)*100:.2f}%)")

# 實體分析
def analyze_entities(df):
    print("\n" + "="*50)
    print("實體分析")
    print("="*50)
    
    # 實體類型分析
    entity_type_counts = df['ENTITY_TYPE'].value_counts()
    print("\n實體類型分佈：")
    for entity_type, count in entity_type_counts.items():
        print(f"{entity_type}: {count:,} 筆 ({count/len(df)*100:.2f}%)")
    
    # 前 20 個最常出現的實體
    top_entities = df['ENTITY_NAME'].value_counts().head(20)
    print("\n前 20 個最常出現的實體：")
    for entity, count in top_entities.items():
        print(f"{entity}: {count:,} 筆 ({count/len(df)*100:.2f}%)")
    
    # 國家分佈
    country_counts = df['COUNTRY_CODE'].value_counts()
    print(f"\n總共涵蓋 {len(country_counts)} 個國家/地區")
    print("\n前 15 個國家/地區：")
    for country, count in country_counts.head(15).items():
        if pd.notna(country):  # 檢查是否為 NaN
            print(f"{country}: {count:,} 筆 ({count/len(df)*100:.2f}%)")
    
    # 缺失國家代碼
    missing_country = df['COUNTRY_CODE'].isnull().sum()
    print(f"\n缺失國家代碼的記錄數：{missing_country} ({missing_country/len(df)*100:.2f}%)")

# 情緒分析
def analyze_sentiment(df):
    print("\n" + "="*50)
    print("情緒分析")
    print("="*50)
    
    # 相關性分佈
    relevance_stats = df['RELEVANCE'].describe()
    print("\n相關性 (RELEVANCE) 統計：")
    for stat, value in relevance_stats.items():
        print(f"{stat}: {value:.2f}")
    
    # 情緒分數分佈
    sentiment_stats = df['EVENT_SENTIMENT_SCORE'].dropna().describe()
    print("\n情緒分數 (EVENT_SENTIMENT_SCORE) 統計：")
    for stat, value in sentiment_stats.items():
        print(f"{stat}: {value:.4f}")
    
    # CSS分數分佈
    css_stats = df['CSS'].describe()
    print("\nCSS (Composite Sentiment Score) 統計：")
    for stat, value in css_stats.items():
        print(f"{stat}: {value:.4f}")
    
    # 情緒分數分佈圖
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['EVENT_SENTIMENT_SCORE'].dropna(), bins=50, alpha=0.7)
    plt.title('EVENT_SENTIMENT_SCORE 分佈')
    plt.xlabel('分數')
    plt.ylabel('頻率')
    
    plt.subplot(1, 2, 2)
    plt.hist(df['CSS'], bins=50, alpha=0.7)
    plt.title('CSS 分佈')
    plt.xlabel('分數')
    plt.ylabel('頻率')
    
    plt.tight_layout()
    plt.savefig('ravenpack_sentiment_distribution.png')
    
    # 情緒極性分佈
    positive_sentiment = (df['EVENT_SENTIMENT_SCORE'] > 0).sum()
    negative_sentiment = (df['EVENT_SENTIMENT_SCORE'] < 0).sum()
    neutral_sentiment = (df['EVENT_SENTIMENT_SCORE'] == 0).sum()
    missing_sentiment = df['EVENT_SENTIMENT_SCORE'].isnull().sum()
    
    print("\n情緒極性分佈 (EVENT_SENTIMENT_SCORE)：")
    print(f"正面情緒 (>0): {positive_sentiment:,} 筆 ({positive_sentiment/(len(df)-missing_sentiment)*100:.2f}% 的有效記錄)")
    print(f"負面情緒 (<0): {negative_sentiment:,} 筆 ({negative_sentiment/(len(df)-missing_sentiment)*100:.2f}% 的有效記錄)")
    print(f"中性情緒 (=0): {neutral_sentiment:,} 筆 ({neutral_sentiment/(len(df)-missing_sentiment)*100:.2f}% 的有效記錄)")
    print(f"缺失情緒: {missing_sentiment:,} 筆 ({missing_sentiment/len(df)*100:.2f}% 的總記錄)")
    
    # 分析 CSS 和其他分數的關係
    columns_to_analyze = ['PEQ', 'BEE', 'BMQ', 'BAM', 'BCA', 'BER']
    print("\nCSS 與其他指標的相關性：")
    correlations = df[['CSS'] + columns_to_analyze].corr()['CSS']
    for col, corr in correlations.items():
        if col != 'CSS':
            print(f"CSS 與 {col} 的相關係數: {corr:.4f}")

# 主題與分類分析
def analyze_topics(df):
    print("\n" + "="*50)
    print("主題與分類分析")
    print("="*50)
    
    # 主題分析
    if 'TOPIC' in df.columns:
        topic_counts = df['TOPIC'].value_counts().head(20)
        print("\n前 20 個主題分佈：")
        for topic, count in topic_counts.items():
            if pd.notna(topic):  # 檢查是否為 NaN
                print(f"{topic}: {count:,} 筆 ({count/(len(df)-df['TOPIC'].isnull().sum())*100:.2f}% 的有效記錄)")
    
    # 群組分析
    if 'GROUP' in df.columns:
        group_counts = df['GROUP'].value_counts().head(20)
        print("\n前 20 個群組分佈：")
        for group, count in group_counts.items():
            if pd.notna(group):  # 檢查是否為 NaN
                print(f"{group}: {count:,} 筆 ({count/(len(df)-df['GROUP'].isnull().sum())*100:.2f}% 的有效記錄)")
    
    # 類型分析
    if 'TYPE' in df.columns:
        type_counts = df['TYPE'].value_counts().head(20)
        print("\n前 20 個類型分佈：")
        for type_val, count in type_counts.items():
            if pd.notna(type_val):  # 檢查是否為 NaN
                print(f"{type_val}: {count:,} 筆 ({count/(len(df)-df['TYPE'].isnull().sum())*100:.2f}% 的有效記錄)")

# 時間序列情緒分析
def analyze_sentiment_time_series(df):
    print("\n" + "="*50)
    print("時間序列情緒分析")
    print("="*50)
    
    # 確保日期格式正確
    df['DATE'] = pd.to_datetime(df['RPA_DATE_UTC'])
    
    # 計算每日平均情緒
    daily_avg_sentiment = df.groupby('DATE')['EVENT_SENTIMENT_SCORE'].mean()
    daily_avg_css = df.groupby('DATE')['CSS'].mean()
    
    print(f"\n每日平均情緒分數範圍：{daily_avg_sentiment.min():.4f} 至 {daily_avg_sentiment.max():.4f}")
    print(f"每日平均 CSS 範圍：{daily_avg_css.min():.4f} 至 {daily_avg_css.max():.4f}")
    
    # 繪製時間序列情緒圖
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    daily_avg_sentiment.plot()
    plt.title('每日平均情緒分數 (EVENT_SENTIMENT_SCORE)')
    plt.xlabel('日期')
    plt.ylabel('平均分數')
    
    plt.subplot(2, 1, 2)
    daily_avg_css.plot()
    plt.title('每日平均 CSS 分數')
    plt.xlabel('日期')
    plt.ylabel('平均分數')
    
    plt.tight_layout()
    plt.savefig('ravenpack_sentiment_timeseries.png')
    
    # 計算每週平均情緒
    df['WEEK'] = df['DATE'].dt.to_period('W')
    weekly_avg_sentiment = df.groupby('WEEK')['EVENT_SENTIMENT_SCORE'].mean()
    weekly_avg_css = df.groupby('WEEK')['CSS'].mean()
    
    print(f"\n每週平均情緒分數範圍：{weekly_avg_sentiment.min():.4f} 至 {weekly_avg_sentiment.max():.4f}")
    print(f"每週平均 CSS 範圍：{weekly_avg_css.min():.4f} 至 {weekly_avg_css.max():.4f}")
    
    # 企業與情緒分析
    if len(df['ENTITY_TYPE'].unique()) > 1:
        print("\n不同實體類型的平均情緒分數：")
        entity_type_sentiment = df.groupby('ENTITY_TYPE')['EVENT_SENTIMENT_SCORE'].mean().sort_values()
        entity_type_css = df.groupby('ENTITY_TYPE')['CSS'].mean().sort_values()
        
        for entity_type, mean_sentiment in entity_type_sentiment.items():
            css = entity_type_css[entity_type]
            print(f"{entity_type}: EVENT_SENTIMENT_SCORE={mean_sentiment:.4f}, CSS={css:.4f}")

# 情緒與回報率的潛在關係
def analyze_sentiment_return_potential(df):
    print("\n" + "="*50)
    print("潛在股票回報關係分析")
    print("="*50)
    
    # 分析企業實體的情緒分佈
    company_entities = df[df['ENTITY_TYPE'] == 'COMP']
    if len(company_entities) > 0:
        print(f"\n企業實體數量: {len(company_entities)}")
        
        # 計算每間公司的平均情緒
        company_sentiment = company_entities.groupby('ENTITY_NAME')['EVENT_SENTIMENT_SCORE'].mean().sort_values()
        company_css = company_entities.groupby('ENTITY_NAME')['CSS'].mean().sort_values()
        
        print("\n情緒最負面的 10 間公司：")
        for company, sentiment in company_sentiment.head(10).items():
            css = company_css.get(company, np.nan)
            print(f"{company}: EVENT_SENTIMENT_SCORE={sentiment:.4f}, CSS={css:.4f}")
        
        print("\n情緒最正面的 10 間公司：")
        for company, sentiment in company_sentiment.tail(10).items():
            css = company_css.get(company, np.nan)
            print(f"{company}: EVENT_SENTIMENT_SCORE={sentiment:.4f}, CSS={css:.4f}")
    
    # 分析不同情緒分數區間的分佈
    print("\n情緒分數區間分佈：")
    score_bins = [-1.0, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1.0]
    bin_labels = [f"{score_bins[i]:.1f} to {score_bins[i+1]:.1f}" for i in range(len(score_bins)-1)]
    
    df['SENTIMENT_BINS'] = pd.cut(df['EVENT_SENTIMENT_SCORE'], bins=score_bins, labels=bin_labels)
    sentiment_bin_counts = df['SENTIMENT_BINS'].value_counts().sort_index()
    
    for bin_range, count in sentiment_bin_counts.items():
        if pd.notna(bin_range):  # 檢查是否為 NaN
            print(f"{bin_range}: {count:,} 筆 ({count/(len(df)-df['EVENT_SENTIMENT_SCORE'].isnull().sum())*100:.2f}% 的有效記錄)")

# 執行完整分析
def run_full_analysis():
    print("="*80)
    print("RavenPack 2000.pqt 資料詳細分析")
    print("="*80)
    
    # 載入資料
    df = load_and_analyze_ravenpack()
    if df is None:
        return
    
    # 執行各項分析
    explore_basic_info(df)
    analyze_time_distribution(df)
    analyze_entities(df)
    analyze_sentiment(df)
    analyze_topics(df)
    analyze_sentiment_time_series(df)
    analyze_sentiment_return_potential(df)
    
    print("\n" + "="*50)
    print("分析完成！圖表已保存至當前目錄。")
    print("="*50)

# 主程式
if __name__ == "__main__":
    run_full_analysis()