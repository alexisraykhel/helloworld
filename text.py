import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class NLP():
    def __init__(self,train_df,test_df,subset=None):
        if subset is None:
            self.train_df = train_df
            self.test_df = test_df
        else:
            self.train_df = train_df[:subset]
            self.test_df = test_df[:subset]

    def convert_descriptions(self):
        train_descriptions = self.train_df['description']
        test_descriptions = self.test_df['description']

        def print_some(descriptions, num):
            count = 0
            for desc in descriptions:
                if count == num:
                    break
                else:
                    print("desc: ", desc)
                    count +=1

        def sentiment(descriptions, sentiment):
            analyzer = SentimentIntensityAnalyzer()
            with_sentiment = []
            for sentence in descriptions:
                sentence = sentence.replace('<br />', '')
                sentence = sentence.replace('<p>', '')
                sentence = sentence.replace('<a  website_redacted', '')
                vs = analyzer.polarity_scores(sentence)
                #print("{:-<65} {}".format(sentence, str(vs)))
                with_sentiment.append(vs.get(sentiment))

            return with_sentiment

        def lengths(descriptions):
            converted = []
            for sentence in descriptions:
                one = len(sentence)
                converted.append(one)
            return converted

        converted_train = []
        for str in train_descriptions:
            converted = [len(str.split()),len(str)]
            converted_train.append(converted)

        converted_test = []
        for str in test_descriptions:
            converted = [len(str.split()),len(str)]
            converted_test.append(converted)

        compound_train_sent = np.vstack((sentiment(train_descriptions, 'compound')))
        compound_test_sent = np.vstack((sentiment(test_descriptions, 'compound')))
        length_train = np.vstack((lengths(train_descriptions)))
        length_test = np.vstack((lengths(test_descriptions)))

        print("NLP Shapes: ", compound_train_sent.shape,compound_test_sent.shape)

        return compound_train_sent, compound_test_sent, length_train, length_test
