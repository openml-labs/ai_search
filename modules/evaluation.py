# This is a work in progress and nothing here is confirmed to work properly yet


def get_common_words(results, config):
    evaluation_df = results.copy()
    # Remove stopwords
    import nltk

    nltk.download("stopwords")
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))

    def remove_stopwords(text):
        return " ".join([word for word in text.split() if word not in stop_words])

    evaluation_df["description_no_stopwords"] = evaluation_df["description"].apply(
        lambda x: remove_stopwords(x)
    )
    query_no_stopwords = remove_stopwords(config["query"])
    from collections import Counter

    counter1 = Counter(query_no_stopwords.split())
    counter2 = Counter(evaluation_df["description_no_stopwords"].values[10].split())
    counter1, counter2
    # Find number of common words between query and description
    common_words = counter1 & counter2
    # count of common words
    sum(common_words.values())
