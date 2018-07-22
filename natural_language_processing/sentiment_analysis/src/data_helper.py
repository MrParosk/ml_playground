import re
import pandas as pd
from nltk.corpus import stopwords
import nltk
from src.misc import fetch_contractions
from bs4 import BeautifulSoup 


CACHE_ENGLISH_STOPWORDS = stopwords.words('english')


def extract_data(file_name, n_rows=None):
    """
    Loading data and putting it into two lists, text and reviews (1-5)
    Arguments:
    file_name -- direction to the data file
    n_rows -- int determine how many rows should be loaded. Default None, i.e. all rows are loaded

    Returns:
    texts -- list of the text reviews
    scores -- list of review scores, 0-4
    """
    df = pd.read_csv(file_name, nrows=n_rows)

    scores = df["Score"].values
    scores = scores > 3
    scores = scores.astype(int)

    texts = df["Text"].values

    return texts, scores


def clean_text(text, contractions, remove_stopwords=True):
    """
    Cleaning a text from contractions and stopwords
    Arguments:
        text -- a string of text
        contractions -- dictionary containing contractions and there corresponding meaning
        remove_stopwords -- boolean declaring wheter we should remove stopwords from the text

    Returns:
        text -- cleaned text
    """
    # Removing HTML tags
    text = BeautifulSoup(text).get_text().lower() 

    # Replace contractions with their longer forms, see above
    text = text.split()
    new_text = []
    for word in text:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)

    text = " ".join(new_text)

    # Keeping letters
    text = re.sub("[^a-z]"," ",text)  

    # Removing stopwords
    if remove_stopwords:
        text = text.split()
        text = [w for w in text if not w in CACHE_ENGLISH_STOPWORDS]
        text = " ".join(text)

    return text


def clean_data(texts, remove_stopwords=True):
    """
    Cleaning up the data set containing text
    Arguments:
        texts -- list of text
        remove_stopwords -- boolean declaring wheter we should remove stopwords from the texts

    Returns:
        texts -- cleaned version of texts
    """

    nltk.download('stopwords')
    contractions = fetch_contractions()

    for i, text in enumerate(texts):
        texts[i] = clean_text(text, contractions, remove_stopwords)

    return texts


def calc_num_words(texts, threshold=1):
    """
    Counting the number of words with a frequency higher than a threshold.
    Arguments:
        texts -- list of text
        threshold -- Count the words with a frequency higher than threshold

    Returns:
        num_words -- number of words with a frequency higher than threshold
    """

    count_dict = {}
    for sentence in texts:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1

    num_words = 0
    for key in count_dict:
        if count_dict[key] > threshold:
            num_words += 1
    
    return num_words


def sentiment(score):
    """
    Coverting probability to sentiment
    Arguments:
        score -- probability of positive sentiment
    """

    if score > 0.5:
        return "positive"
    else:
        return "negative"
