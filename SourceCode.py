# RUN THIS CODE IN GOOGLE COLAB FOR CORRECT EXECUTION 


#%pip install vaderSentiment
#%pip install pandas
#%pip install matplotlib
#%pip install seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from google.colab import files
uploaded = files.upload()



try:
    df = pd.read_excel('1presidential_speeches_with_metadata.xlsx')
    display(df.head())
except FileNotFoundError:
    print("Error: File '1presidential_speeches_with_metadata.xlsx' not found.")
    df = None
except Exception as e:
    print(f"An error occurred: {e}")
    df = None

# Data Dimensions and Types
print("Data Dimensions:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\nData Types:")
print(df.dtypes)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())
missing_percentage = (df.isnull().sum() / len(df)) * 100
print("\nPercentage of Missing Values:")
print(missing_percentage)

# Descriptive Statistics
print("\nDescriptive Statistics:")
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        print(f"\nColumn: {col}")
        print(df[col].describe())
    else:
        print(f"\nColumn: {col}")
        print(f"Unique values: {df[col].nunique()}")
        print(df[col].value_counts())

# Potential Outliers (for numerical columns)
print("\nPotential Outliers (using IQR for numerical columns):")
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"\nColumn: {col}")
        print(f"Number of outliers: {len(outliers)}")
        if not outliers.empty:
            print("Outlier values:")
            print(outliers[col])


# Load the dataset (update the path as needed)
df = pd.read_excel("1presidential_speeches_with_metadata.xlsx")  # or read_excel()

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Apply sentiment analysis to each speech and add scores
df['compound'] = df['speech'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

# Classify based on compound score
def classify_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['compound'].apply(classify_sentiment)
df.head()


# Frequency distribution of categorical variables
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
df['President'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Frequency of Speeches per President')
plt.xlabel('President')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
df['Party'].value_counts().plot(kind='bar', color='lightcoral')
plt.title('Frequency of Speeches per Party')
plt.xlabel('Party')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
df['Vice President'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Frequency of Speeches per Vice President')
plt.xlabel('Vice President')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Temporal distribution of speeches
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
plt.figure(figsize=(10, 5))
sns.countplot(x='year', data=df, palette='viridis')
plt.title('Number of Speeches per Year')
plt.xlabel('Year')
plt.ylabel('Number of Speeches')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Relationship between Party and speech content (simplified example)
# (More advanced techniques like word clouds or topic modeling would require additional libraries.)
party_speech_lengths = df.groupby('Party')['speech'].apply(lambda x: x.str.len().mean())
plt.figure(figsize=(8, 6))
party_speech_lengths.plot(kind='bar', color='orange')
plt.title('Average Speech Length by Party')
plt.xlabel('Party')
plt.ylabel('Average Speech Length')
plt.show()

# Relationship between President and speech length
df['speech_length'] = df['speech'].str.len()
plt.figure(figsize=(12, 6))
sns.boxplot(x='President', y='speech_length', data=df, palette='Set3')
plt.title('Speech Length Distribution by President')
plt.xlabel('President')
plt.ylabel('Speech Length')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Relationship between 'President' and 'Party'
plt.figure(figsize=(10, 6))
sns.countplot(x='President', hue='Party', data=df, palette='Set1')
plt.title('Number of Speeches by President and Party')
plt.xlabel('President')
plt.ylabel('Number of Speeches')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Distribution of speech lengths
plt.figure(figsize=(10, 6))
sns.histplot(df['speech_length'], kde=True, color='skyblue')
plt.title('Distribution of Speech Lengths')
plt.xlabel('Speech Length')
plt.ylabel('Frequency')
plt.show()

# Relationship between year and speech length
plt.figure(figsize=(12,6))
sns.regplot(x='year', y='speech_length', data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Relationship between Year and Speech Length')
plt.xlabel('Year')
plt.ylabel('Speech Length')
plt.show()

# Improve the existing plot of Number of Speeches per Year
plt.figure(figsize=(10, 5))
sns.countplot(x='year', data=df, palette='viridis')
plt.title('Number of Speeches per Year')
plt.xlabel('Year')
plt.ylabel('Number of Speeches')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
