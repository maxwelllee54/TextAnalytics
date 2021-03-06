##############################################################################################
# Filename: ReviewAnalyzer.py                                                                #
# Date: Feb 22nd, 2017                                                                       #
# Author: Yanzhe Li                                                                          #
##############################################################################################
import nltk
import re
import pandas as pd

class reviewAnalyzer():
    '''
    a class function to process the data frame by following steps:
        1. Use NLTK to tokenize text in the "review text" column
        2. Give the user options for "bag-of-words", "bag-of-words with stemming and stop words removal",
           "POS with all nouns" and "POS with only NNP" by customized functions
        3. Upon choosing the functions, add top 30 features to dataframe as new columns for further analysis and return a new dataframe
    '''

    def __init__(self, df, top_features=30):
        '''

        :param df: the original input data
        :param ngrams: options for n-grams bag of words
        :param top_features: options for number of top features returned
        '''

        self.df = df
        # rename column names to uppercases to avoid ambiguity that the feature words may be same as column names
        self.df.columns = map(str.upper, df.columns)
        self.top_features = top_features

    def simple_bags(self, ngrams=1):
        '''
        :return: Use a simple bag of words to analyze
        '''

        data = self.df.copy()

        for ind in range(len(data)):
            rawText = data.loc[ind, 'REVIEW TEXT']
            # Use regular expressions to get a pure letter text and transfer all words to lower case
            letterText = re.sub('[^a-zA-Z]', ' ', rawText).lower()
            tokens = nltk.word_tokenize(letterText, language='english')
            text = []
            for n in range(1, ngrams + 1):
                tempText = nltk.ngrams(tokens, n)
                text.extend([' '.join(words).strip() for words in tempText])

            freq = nltk.FreqDist(text)
            sorted_freq = sorted(freq.items(), key=lambda k: k[1], reverse=True)[:self.top_features]

            for feature, freqency in sorted_freq:
                data.loc[ind, feature] = freqency

        data = data.fillna(0)
        return data

    def bag_of_words_stem_stop(self, ngrams=1):
        '''
        :return: Use a bag of words approach with stemming and stop words removal to analyze
        '''

        data = self.df.copy()

        stopwords = nltk.corpus.stopwords.words('english')
        wnl = nltk.WordNetLemmatizer()

        for ind in range(len(data)):
            rawText = data.loc[ind, 'REVIEW TEXT']
            # Use regular expressions to get a pure letter text and transfer all words to lower case
            letterText = re.sub('[^a-zA-Z]', ' ', rawText).lower()
            tokens = nltk.word_tokenize(letterText, language='english')

            text = []
            for n in range(1, ngrams + 1):
                tempText = nltk.ngrams(tokens, n)
                text.extend([' '.join(words).strip() for words in tempText])

            # Remove Stop words and stemming
            noStopWordsText = [words for words in text if words not in stopwords]
            stemmedText = [wnl.lemmatize(words) for words in noStopWordsText]

            freq = nltk.FreqDist(stemmedText)
            sorted_freq = sorted(freq.items(), key=lambda k: k[1], reverse=True)[:self.top_features]

            for feature, freqency in sorted_freq:
                data.loc[ind, feature] = freqency

        data = data.fillna(0)
        return data

    def pos_tags(self, posList=('NN', 'NNP', 'NNS', 'NNPS'), ngrams=1):
        '''
        :return: Use POS approach and focus on all the noun forms (NN, NNP, NNS, NNPS)
        '''

        data = self.df.copy()
        print(posList)

        for ind in range(len(data)):
            rawText = data.loc[ind, 'REVIEW TEXT']
            # Use regular expressions to get a pure letter text and transfer all words to lower case
            tokens = nltk.word_tokenize(rawText, language='english')

            text = []
            for n in range(1, ngrams+1):
                tempText = nltk.ngrams(tokens, n)
                text.extend([' '.join(words).strip() for words in tempText])

            # pos tags the tokens
            posTag = [(pos, tag) for (pos, tag) in nltk.pos_tag(tokens=text) if tag in posList]

            freq = nltk.FreqDist(posTag)
            sorted_freq = sorted(freq.items(), key=lambda k: k[1], reverse=True)[:self.top_features]

            for feature, freqency in sorted_freq:
                #print(feature, '*****', freqency)

                data.loc[ind, feature[0]] = freqency

        data = data.fillna(0)
        return data


if __name__ == '__main__':
    df = pd.read_csv('fashion_data.csv')
    analyzer = reviewAnalyzer(df)
    # test functions
    #simple_bags = analyzer.simple_bags(ngrams=1)
    posTagNNP = analyzer.pos_tags()
    print(posTagNNP.iloc[:, 7:].sum().sort_values(ascending=False)[:30])