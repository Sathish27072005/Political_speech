🗳️ U.S. Presidential Speech Sentiment Analysis
This project analyzes the sentiment of U.S. presidential speeches using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool. It is part of a broader data analytics project focused on political communication and public sentiment.

📓 Project Overview
Goal: Analyze sentiment trends across different U.S. presidents' speeches.

Tool: VADER Sentiment Analysis from the NLTK library.

Data: A dataset of presidential speeches over several decades.

Output: Visualization and analysis of sentiment polarity over time and across presidents.

🧪 Contents
main.ipynb: The core Jupyter Notebook containing all data preprocessing, sentiment scoring, and visualization logic.

data/: (Not included here) Directory expected to contain the presidential speech dataset.

🔧 Setup Instructions
Clone the repository:

bash
Copy
Edit
git clone https://github.com/Sathish27072005/Political_speech
Install dependencies:


pip install -r requirements.txt

Run the notebook:

google colab : main.ipynb

📊 Methodology
Text Preprocessing: Lowercasing, tokenization, and optional stopword removal.

Sentiment Scoring: Each sentence in a speech is scored using VADER.

Aggregation: Sentiment scores are averaged per speech and per president.

Visualization: Line plots and bar charts display sentiment trends.

📈 Sample Visuals
Average sentiment over time.

Comparison of sentiment between presidents.

Distribution of sentiment polarity scores.

🧰 Dependencies
Python 3.x

nltk

matplotlib

pandas

seaborn

numpy

vaderSentiment

(Include a requirements.txt file to make setup easier.)

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for suggestions and improvements.
