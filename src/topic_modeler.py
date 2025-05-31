import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class TopicModeler:
    def __init__(self, num_topics=5, max_features=500, ngram_range=(1, 1)):
        self.num_topics = num_topics
        self.vectorizer = CountVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=ngram_range
        )
        self.lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            learning_method='batch',
            random_state=42
        )
        self.fitted = False

    class Preprocessor:
        @staticmethod
        def clean_text(text):
            text = text.lower()
            text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation/numbers
            return text

        @staticmethod
        def clean_series(text_series):
            return text_series.fillna('').apply(TopicModeler.Preprocessor.clean_text)

    def fit(self, text_series: pd.Series):
        """
        Fit LDA topic model on cleaned text headlines.
        """
        cleaned = self.Preprocessor.clean_series(text_series)
        self.vectorized = self.vectorizer.fit_transform(cleaned)
        self.lda_model.fit(self.vectorized)
        self.fitted = True
        return self.get_topics()

    def get_topics(self, n_top_words=10):
        """
        Return top words per topic.
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call `fit()` first.")

        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for idx, topic in enumerate(self.lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-n_top_words:][::-1]]
            topics.append((f"Topic {idx + 1}", top_words))
        return topics
