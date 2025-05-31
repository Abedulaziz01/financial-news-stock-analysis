class Analyzer:
    @staticmethod
    def describe_headline_lengths(df):
        """Returns basic descriptive stats for headline lengths."""
        return df['headline_length'].describe()

    @staticmethod
    def count_articles_by_publisher(df, top_n=10):
        """Counts the number of articles per publisher."""
        return df['publisher'].value_counts().head(top_n)

    @staticmethod
    def publication_frequency_by_day(df):
        """Analyzes how often articles are published per day."""
        df['date_only'] = df['date'].dt.date
        return df.groupby('date_only').size()

    @staticmethod
    def publishing_hours(df):
        """Returns distribution of articles across hours of the day."""
        df['hour'] = df['date'].dt.hour
        return df['hour'].value_counts().sort_index()