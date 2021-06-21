import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk import TweetTokenizer
import matplotlib.pyplot as plt

'''
import flair  # installing flair leads to issues with CUDA!
'''
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.inspection import permutation_importance
import regex as re
import warnings
from sklearn.feature_selection import SelectFromModel

warnings.filterwarnings("ignore")
np.random_state = 0


class FeatureEngineer:
    """
    A class to manually extract multiple features from textual data.

    sentiment feature extraction inspired by:
    https://medium.com/@b.terryjack/nlp-pre-trained-sentiment-analysis-1eb52a9d742c, last access: 31.05.2021, 8:13pm
    """

    def __init__(self):
        """
        Constructor.
        """
        self.tweet_tokenizer = TweetTokenizer()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        '''
        self.flair_analyzer = flair.models.TextClassifier.load("en-sentiment")
        '''

    @staticmethod
    def sequence_len(text_data):
        """
        Extracts the total number of characters from a pd.Series of textual data.

        :param pd.Series text_data: the textual components of the memes
        :return: a pd.Series of the same size as the input Series.
        Each entry holds the sequence length of the corresponding
        entry in the input Series.
        """

        def extract_len(sequence):
            """
            Calculates the length of one sequence.

            :param str sequence: a text sequence
            :return: the length of the text sequence
            """
            return len(sequence)

        sequence_lengths = text_data.apply(extract_len)
        sequence_lengths.name = "seq_lengths"
        return sequence_lengths

    @staticmethod
    def punctuation(text_data):
        """
        Extracts the total number of the punctuation chars ,.!?

        :param text_data: pd.Series text_data: the textual components of the memes
        :return: a pd.Dataframe containing one column for each extracted feature
        """

        def period(sequence):
            """
            Counts the number of periods in a given sequence.

            :param str sequence: The textual component of a meme.
            :return: number of periods in the sequence
            """
            return re.subn(r"\.", '', sequence)[1]

        def comma(sequence):
            """
            Counts the number of commas in a given sequence.

            :param str sequence: The textual component of a meme.
            :return: number of commas in the sequence
            """
            return re.subn(r"\,", '', sequence)[1]

        def question_mark(sequence):
            """
            Counts the number of question_marks in a given sequence.

            :param str sequence: The textual component of a meme.
            :return: number of question_marks in the sequence
            """
            return re.subn(r"\?", '', sequence)[1]

        def exclamation_mark(sequence):
            """
            Counts the number of exclamation_marks in a given sequence.

            :param str sequence: The textual component of a meme.
            :return: number of exclamation_marks in the sequence
            """
            return re.subn(r"\!", '', sequence)[1]

        periods = text_data.apply(func=period)
        periods.name = "num_periods"
        commas = text_data.apply(func=comma)
        commas.name = "num_commas"
        question_marks = text_data.apply(func=question_mark)
        question_marks.name = "num_question_marks"
        exclamation_marks = text_data.apply(func=exclamation_mark)
        exclamation_marks.name = "num_exclamation_marks"
        return pd.concat([periods, commas, question_marks, exclamation_marks], axis=1)

    def simple_bow(self, text_data, all_tokens=None):
        """
        Extracts a bag of words representation of a pd.Series containing sequences.

        :param pd.Series text_data: contains the text sequences representing the textual components of memes.
        :param pd.Series all_tokens: contains all tokens occuring in the textual data, if None, it's created on
        the given data
        :return: a dictionary having the keys "features" and "all_tokens". The value of "features" is a pd.DataFrame
        containing the extracted features. "all_tokens" has a pd.Series of all discovered tokens as its value.
        """

        def tokenize_sequence(sequence):
            """
            Tokenizes a sequence of text using nltk.TweetTokenizer.

            :param str sequence: a sequence of text that has to be tokenized
            :return: a list containing strings. Each string is one token found in the sequence.
            """
            return self.tweet_tokenizer.tokenize(text=sequence)

        def count_tokens(all_tokens):
            """
            Checks whether a token in all_tokens is contained in the sequences.

            :param pd.Series all_tokens: contains all tokens occurring in the textual data, if None, it's created on
            the given data
            :return: a pd.DataFrame having len(all_tokens) many columns. It contains the information whether a
            token appears in a sequence or not.
            """

            def check_token(sequence, token):
                """
                Checks whether a token occurs in a sequence.

                :param str sequence:  a text sequence.
                :param str token: a token
                :return: 1 if the token is in the sequence, 0 otherwise
                """
                if token in sequence:
                    return 1
                else:
                    return 0

            all_features = []
            for token in all_tokens:
                token_in_seq = text_data.apply(func=check_token, token=token)
                token_in_seq.name = token
                all_features.append(token_in_seq)
            return pd.concat(all_features, axis=1)

        if all_tokens is None:  # if performed on training data
            all_tokens = []

            def collapse_sequences(tokenized_row, all_tokens):
                """
                Combines lists of tokens to one large list of tokens.

                :param list tokenized_row: a list of tokens stored in one row of the column having the textual data.
                :param list all_tokens: a list having all already concatenated tokens
                """
                all_tokens += tokenized_row

            tokenized_train = text_data.apply(func=tokenize_sequence)
            tokenized_train.apply(func=collapse_sequences, all_tokens=all_tokens)
            all_tokens = pd.Series(all_tokens).drop_duplicates()
            all_tokens.index = all_tokens.values
            all_tokens = all_tokens.drop([",", ".", "!", "?", "[", "]"]).values  # for xgboost & redundancy avoidance
            return {"features": count_tokens(all_tokens=all_tokens), "all_tokens": all_tokens}
        else:
            return {"features": count_tokens(all_tokens=all_tokens), "all_tokens": all_tokens}

    @staticmethod
    def textblob_sequence_sentiment(text_data):
        """
        Estimates the sequence sentiment using TextBlob.

        :param pd.Series text_data: a Series containing the sequences
        :return: a pd.Series containing the estimated sequence sentiments.
        """

        def sentiment(sequence):
            """
            Estimates the sentiment of one specific sequence using TextBlob.

            :param str sequence: a text sequence
            :return: the estimated sentiment
            """
            return TextBlob(sequence).sentiment[0]

        textblob_sequence_sent = text_data.apply(func=sentiment)
        textblob_sequence_sent.name = "textblob_sequence_sentiment"
        return textblob_sequence_sent

    @staticmethod
    def textblob_sequence_subjectivity(text_data):
        """
        Estimates the sequence subjectivity using TextBlob.

        :param pd.Series text_data: a Series containing the sequences
        :return: the estimated subjectivity
        """

        def subjectivity(sequence):
            """
            Estimates the subjectivity of one specific sequence using TextBlob.

            :param str sequence: a text sequence
            :return: the estimated subjectivity
            """
            return TextBlob(sequence).sentiment[1]

        textblob_sequence_sub = text_data.apply(func=subjectivity)
        textblob_sequence_sub.name = "textblob_sequence_subjectivity"
        return textblob_sequence_sub

    def num_tweet_tokens(self, text_data):
        """
        Extracts the number of tokens per sequence using nltk.TweetTokenizer.

        :param pd.Series text_data: contains the text sequences of all observations
        :return: a pd.Series containing the number of tokens per sequence
        """
        tweet_tokenizer = self.tweet_tokenizer

        def extract_num_tokens(sequence):
            """
            Calculates the length of a sequence.

            :param str sequence: a sequence of text
            :return: the length of the sequence
            """
            return len(tweet_tokenizer.tokenize(text=sequence))

        num_tweet_tokens = text_data.apply(func=extract_num_tokens)
        num_tweet_tokens.name = "num_tweet_tokens"
        return num_tweet_tokens

    def vader_sequence_sentiment(self, text_data):
        """
        Estimates the sequence sentiment using vader.

        :param pd.Series text_data: a Series containing the sequences
        :return: a pd.Series containing the estimated sequence sentiments.
        """
        analyzer = self.vader_analyzer

        def seq_sentiment(sequence):
            """
            Estimates the sentiment of one specific sequence using vader.

            :param str sequence: a text sequence
            :return: the estimated sentiment
            """
            return analyzer.polarity_scores(text=sequence)["compound"]

        vader_sequence_sent = text_data.apply(func=seq_sentiment)
        vader_sequence_sent.name = "vader_sequence_sentiment"
        return vader_sequence_sent

    def textblob_tokenwise_sentiment(self, text_data):
        """
        Estimates the token sentiments using TextBlob. Only the most negative and the most positive are stored.

        :param pd.Series text_data: a Series containing the sequences
        :return: the estimated extreme token sentiments
        """
        tweet_tokenizer = self.tweet_tokenizer

        def worst_sentiment(sequence):
            """
            Extracts the worst token sentiment.

            :param str sequence: a text sequence
            :return: the worst sentiment a token in the sequence achieves
            """
            tokens = tweet_tokenizer.tokenize(text=sequence)
            worst = 0
            for token in tokens:
                sentiment = TextBlob(token).sentiment[0]  # entry 0 contains sentiment/polarity, 1 contains subjectivity
                if sentiment < worst:
                    worst = sentiment
            return worst

        def best_sentiment(sequence):
            """
            Extracts the best token sentiment.

            :param str sequence: a text sequence
            :return: the best sentiment a token in the sequence achieves
            """
            tokens = tweet_tokenizer.tokenize(text=sequence)
            best = 0
            for token in tokens:
                sentiment = TextBlob(token).sentiment[0]
                if sentiment > best:
                    best = sentiment
            return best

        worst_textblob_token_sent = text_data.apply(func=worst_sentiment)
        worst_textblob_token_sent.name = "worst_textblob_token_sentiment"
        best_textblob_token_sent = text_data.apply(func=best_sentiment)
        best_textblob_token_sent.name = "best_textblob_token_sentiment"
        return pd.concat([best_textblob_token_sent, worst_textblob_token_sent], axis=1)

    '''
    def flair_sequence_sentiment(self, text_data):
        """
        Estimates the sequence sentiment using flair.
        Flair leads to issues with cuda. Thus it can't be used with all environments.

        :param pd.Series text_data: a Series containing the sequences
        :return: a pd.Series containing the estimated sequence sentiments.
        """
        flair_analyzer = self.flair_analyzer

        def sentiment(sequence):
            """
            Estimates the sentiment of one specific sequence using flair.

            :param str sequence: a text sequence
            :return: the estimated sentiment
            """
            flair_seq = flair.data.Sentence(sequence)
            flair_analyzer.predict(flair_seq)
            sent = str(flair_seq.labels[0])
            kind = sent[:8]
            value = float(sent[10:-1])
            if kind == "NEGATIVE":
                value = value * (-1)
            return value

        flair_sequence_sent = text_data.apply(func=sentiment)
        flair_sequence_sent.name = "flair_sequence_sentiment"
        return flair_sequence_sent
    '''

    def create_all(self, data, write_name, x_col="text", y_col="label", all_tokens=None):
        """
        Extracts all features this class can automatically extract.

        :param pd.DataFrame data: A Dataset containing both, the text sequences and the labels
        :param str x_col: the name of the column containing the text sequences
        :param str y_col: the name of the column containing the class labes
        :param pd.Series all_tokens: contains all tokens found by nltk.TweetTokenizer
        :param bool write_name: defines whether the created features have to be written to a csv file or not.
        Doesn't write if None, writes a file with having the given name otherwise.
        :return: a dictionary having the keys "x", "y", and "all_tokens". The value of the key "x" contains a
        pd.DataFrame having all extracted features. The value of the key "y" is a pd.Series containing the class labels,
        and the value of the key "all_tokens" is a pd.Series containing all tokens found by nltk.TweetTokenizer in
        the text sequences (of the data available at train time)
        """
        text = data[x_col]
        y = data[y_col]
        seq_len = self.sequence_len(text_data=text)
        num_tokens = self.num_tweet_tokens(text_data=text)
        punctuation_count = self.punctuation(text_data=text)
        if all_tokens is None:
            bow_result = self.simple_bow(text_data=text)
            bow = bow_result["features"]
            all_tokens = bow_result["all_tokens"]
        else:
            bow = self.simple_bow(text_data=text, all_tokens=all_tokens)["features"]

        vader = self.vader_sequence_sentiment(text_data=text)
        tb_token = self.textblob_tokenwise_sentiment(text_data=text)
        tb_seq = self.textblob_sequence_sentiment(text_data=text)
        tb_sub = self.textblob_sequence_subjectivity(text_data=text)
        '''
        flair_seq = self.flair_sequence_sentiment(text_data=text)
        '''
        x = pd.concat([seq_len,
                       num_tokens,
                       vader,
                       tb_token,
                       tb_seq,
                       tb_sub,
                       punctuation_count,
                       bow],
                      axis=1)  # add flair_seq if using flair as well

        if not write_name is None:
            pd.concat([x, y], axis=1).to_csv("../../data/manual_features/" + write_name + ".csv")
        return {"x": x, "y": y, "all_tokens": all_tokens}

    def perform_classification(self, clf, model_name, X_train, y_train, X_test, y_test, pre_f_i=False, post_f_i=False):
        """
        Performs classification on the extracted features.

        :param clf: an sklearn.Estimator that can perform classification
        :param str model_name: name of the model to print it
        :param pd.DataFrame X_train: contains all extracted features of the training data
        :param pd.Series y_train: contains all targets of the training data
        :param pd.DataFrame X_test: contains all extracted features of the test data
        :param pd.Series y_test: contains all targets of the test data
        :param bool pre_f_i: if True, calculates feature importances for models having .coef_ or .feature_importanmces_
        using sklearn.feature_selection.SelectFromModel (a simple form of forward selection)
        :param bool post_f_i: if True, PermutationImportance is performed. THis might be extremely time-consuming.
        """
        print("\n" + model_name + ":")
        if pre_f_i:

            def value_to_bool(value):
                """
                Indicates whether a feature is important or not.
                Assumption: A Feature that is important on at least 7/10 Data Subsets is indeed important.

                :param float value: a value between 0 and 10
                :return: Returns True if a given value is bigger than 7.
                """
                if value >= 7:
                    return True
                return False

            print("Performing Preselection...")
            feature_selector = SelectFromModel(estimator=lr)

            # for more robustness: average over 10 non-disjoint random data subsets:
            all_ids = X_train.index
            random_draw_indices = np.random.randint(low=0, high=len(all_ids), size=(10, int(len(all_ids) * 0.5),))
            best_features_mask = np.zeros(len(X_train.columns))
            for subset in random_draw_indices:
                feature_selector.fit(X=X_train.loc[subset, :], y=y_train[subset])
                best_features_mask = best_features_mask + feature_selector.get_support().astype(float)

            total_best_features_mask = pd.Series(best_features_mask).apply(func=value_to_bool).values
            best_features = X_train.columns[total_best_features_mask]
            print("\nTotal number of Features:", len(total_best_features_mask), "\n")
            print("Preselected Features:", len(best_features), "\n")

            X_train = X_train.loc[:, best_features]
            X_test = X_test.loc[:, best_features]

            print("Metrics using these Features:")
            clf.fit(X=X_train, y=y_train)
            preds = clf.predict(X=X_test)
            print("Accuracy on test:", accuracy_score(y_true=y_test, y_pred=preds))
            print("F1 on test:", f1_score(y_true=y_test, y_pred=preds))
            print("Precision on test:", precision_score(y_true=y_test, y_pred=preds))
            print("Recall on test:", recall_score(y_true=y_test, y_pred=preds))

            if post_f_i:
                print("\nPerforming Permutation Importance:")
                importances = permutation_importance(estimator=clf, X=X_test, y=y_test).importances_mean
                importances_mapped = pd.Series(data=importances, index=X_test.columns)
                sorted_importances = importances_mapped.sort_values(ascending=False)
                print("Sorted Feature Importances:\n", sorted_importances, "\n")
                return {"importances": sorted_importances}
        else:
            clf.fit(X=X_train, y=y_train)
            preds = clf.predict(X=X_test)
            print("Accuracy on test:", accuracy_score(y_true=y_test, y_pred=preds))
            print("F1 on test:", f1_score(y_true=y_test, y_pred=preds))
            print("Precision on test:", precision_score(y_true=y_test, y_pred=preds))
            print("Recall on test:", recall_score(y_true=y_test, y_pred=preds))
            if post_f_i:
                print("\nPerforming Permutation Importance:")
                importances = permutation_importance(estimator=clf, X=X_test, y=y_test).importances_mean
                importances_mapped = pd.Series(data=importances, index=X_test.columns)
                sorted_importances = importances_mapped.sort_values(ascending=False)
                print("Sorted Feature Importances:\n", sorted_importances, "\n")
                return {"importances": sorted_importances}


engineer = FeatureEngineer()
# define train and test data
'''
folds = tools.read_folds(read_path="../../data/folds_nlp", prefix="undersampled_stopped_text", test_fold_id=0)
train_folds = folds["available_for_train"]
test_data = folds["test"]
train_data = train_folds[0]
for i in range(1, len(train_folds) - 1):
    pd.concat([train_data, train_folds[i]], axis=0)


# manually engineer the text features
train_engineered = engineer.create_all(data=train_data, write_name="train_data")
X_train = train_engineered["x"]
y_train = train_engineered["y"]
all_tokens = train_engineered["all_tokens"]
test_engineered = engineer.create_all(data=test_data, all_tokens=all_tokens, write_name="test_data")
X_test = test_engineered["x"]
y_test = test_engineered["y"]
'''

# read the data, since it's already created
train_data = pd.read_csv("../../data/manual_features/train_data.csv")
test_data = pd.read_csv("../../data/manual_features/test_data.csv")
X_train = train_data.drop("label", axis=1)
y_train = train_data["label"]
X_test = test_data.drop("label", axis=1)
y_test = test_data["label"]
all_features = pd.Series(data=np.zeros(shape=len(X_train.columns)), index=X_train.columns, name="feature_importances")

# check the performance
lr = LogisticRegression(random_state=0, max_iter=1_000)
'''
engineer.perform_classification(clf=lr,
                                model_name="LogisticRegression",
                                X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test)

dt = DecisionTreeClassifier(random_state=0)
engineer.perform_classification(clf=dt,
                                model_name="DecisionTreeClassifier",
                                X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test)

ada = AdaBoostClassifier(random_state=0)
engineer.perform_classification(clf=ada,
                                model_name="AdaBoostClassifier",
                                X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test)

rf = RandomForestClassifier(random_state=0)
engineer.perform_classification(clf=rf,
                                model_name="RandomForestClassifier",
                                X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test)

xgb = xgboost.XGBClassifier(random_state=0)
engineer.perform_classification(clf=xgb,
                                model_name="XGBClassifier",
                                X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test)

svc = SVC(random_state=0)
engineer.perform_classification(clf=svc,
                                model_name="SVC",
                                X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test)

nb = GaussianNB()
engineer.perform_classification(clf=nb,
                                model_name="GaussianNB",
                                X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test)

weak_learners = [nb, svc, lr, ada]
stacking = StackingClassifier(estimators=weak_learners)
engineer.perform_classification(clf=rf,
                                model_name="StackingClassifier",
                                X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test)
'''

# logistic regression is a very simple model but performs surprisingly well.
# investigate it further:
lr_importances = engineer.perform_classification(clf=lr,
                                                 model_name="LogisticRegression",
                                                 X_train=X_train,
                                                 y_train=y_train,
                                                 X_test=X_test,
                                                 y_test=y_test,
                                                 post_f_i=True,
                                                 pre_f_i=True)["importances"]
lr_top_20_features = lr_importances.sort_values(ascending=False).head(20)
lr_top_20_features.plot(kind="bar")
plt.show()
