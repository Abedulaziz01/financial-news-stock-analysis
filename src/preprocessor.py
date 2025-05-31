class Preprocessor:
    @staticmethod
    def add_headline_length(df):
        df['headline_length'] = df['headline'].str.len()
        return df
