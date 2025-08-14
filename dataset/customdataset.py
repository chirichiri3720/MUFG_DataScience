import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, AutoModelForMaskedLM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from hydra.utils import to_absolute_path
from .dataset import EssayDataset

from tokenizers import AddedToken

# from collections import Counter
import re
import string

# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from spellchecker import SpellChecker
from collections import Counter
# import nltk
# from nltk.corpus import stopwords
# import warnings

class CustomDataset():
    def __init__(
        self,
        seed: int = 42,
        max_len: int = 512,
        batch_size: int = 2,
        test_size: float = 0.2,
        feature_column: str = 'full_text',
        target_column: str = 'score',
        num_labels: int = 6,
        tokenizer_name : str = 'microsoft/deberta-v3-small',
        **kwargs
    ):
        # Load data
        self.train = pd.read_csv(to_absolute_path("datasets/train.csv"))
        self.test = pd.read_csv(to_absolute_path("datasets/test.csv"))

        self.tokenizer = self.get_tokenizer(tokenizer_name)

        self.seed = seed
        self.max_len = max_len
        self.batch_size =  batch_size
        self.test_size = test_size

        # Load data
        self.train = pd.read_csv(to_absolute_path("datasets/train.csv"))
        self.test = pd.read_csv(to_absolute_path("datasets/test.csv"))

        # # データ削減
        # self.train = self.train[:5000]

        self.feature_column = feature_column
        self.target_column = target_column
        self.num_labels = num_labels

        self.id = self.test['essay_id']
    
    def prepare_loaders(self):
        # Prepare datasets
        X_train, X_val, y_train, y_val = train_test_split(
            self.train.drop(columns=[self.target_column]), self.train[self.target_column], test_size=self.test_size, random_state=self.seed
        )

        train_texts = X_train[self.feature_column].tolist()
        val_texts = X_val[self.feature_column].tolist()
        test_texts = self.test[self.feature_column].tolist()

        train_id = X_train['essay_id'].tolist()
        val_id = X_val['essay_id'].tolist()
        test_id = self.test['essay_id'].tolist()

   
        train_dataset = EssayDataset(train_texts, train_id, y_train.tolist(), self.tokenizer, self.max_len)
        val_dataset = EssayDataset(val_texts, val_id, y_val.tolist(), self.tokenizer, self.max_len)
        test_dataset = EssayDataset(test_texts, test_id, tokenizer=self.tokenizer, max_len=self.max_len)

        # Prepare dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        return train_dataset, val_dataset, train_loader, val_loader, test_loader
    
    def get_tokenizer(self, tokenizer_name):
        if tokenizer_name == 'microsoft/deberta-v3-small':
            tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')
            tokenizer.add_tokens([AddedToken("\n", normalized=False)])
            tokenizer.add_tokens([AddedToken(" "*2, normalized=False)])
        elif tokenizer_name == 'bert-base-uncased':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif tokenizer_name == 'roberta-base':
            tokenizer = AutoTokenizer.from_pretrained('roberta-base') 
        elif tokenizer_name == 'distilbert':
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        elif tokenizer_name == 'xlnet':
            tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')
        elif tokenizer_name == 'albert':
            tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
        elif tokenizer_name == 'electra':
            tokenizer = AutoTokenizer.from_pretrained('google/electra-base-discriminator')
        else:
            raise KeyError(f"{tokenizer_name} is not defined.")
        return tokenizer
    
    def add_prompts(self):
        ...

class V0(CustomDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class V1(CustomDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_prompts()
    
    def add_prompts(self):
        prompts = [
            "Instruction: Evaluating the text and calculating content and wording score. Text: "
        ]
        self.train[self.feature_column] = prompts[0] + self.train[self.feature_column]
        self.test[self.feature_column] = prompts[0] + self.test[self.feature_column]

class V2(CustomDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spell = SpellChecker()
        custom_words = [
            # 省略形
            "isn't", "aren't", "can't", "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't",
            "he'd", "he'll", "he's", "i'd", "i'll", "i'm", "i've", "it's", "let's", "mightn't", "mustn't", "shan't",
            "she'd", "she'll", "she's", "shouldn't", "that's", "there's", "they'd", "they'll", "they're", "they've",
            "we'd", "we're", "we've", "weren't", "what'll", "what's", "where's", "who'll", "who's", "won't",
            "wouldn't", "you'd", "you'll", "you're", "you've",
            # ドメイン固有の用語
            "machine learning", "neural network", "genome", "vaccine", "coronavirus", "covid",
            # 固有名詞
            "microsoft", "google", "python", "new york", "tesla", "facebook",
            # 略語および頭字語
            "nasa", "fbi", "ai", "http", "api", "json", "html", "css",
            # スラングおよび口語表現
            "gonna", "wanna", "cool", "awesome", "lol", "omg", "btw",
            # 未定義検出単語
            'driverless', 'nasa', 'venus', 'dont', 'alot', 'thats', 'europe', 'paris', 'dr', 'american', 'mona',
             'fahrenheit', 'lisa', 'huang', 'luke', 'bogota', 'google', "venus's", '3d', 'im', 'cydonia', 'facs', 
             'vauban', "nasa's", 'carfree', 'april', 'wouldnt', 'earthlike', 'etc', 'americans', 'becuase', 'didnt', 
             'doesnt', 'isnt', 'columbia', 'idaho', 'florida', 'california', 'theres','garvin', 'walter', 
             'monday', 'rosenthal',  'venusian', 'gps', 'bmw', 'heidrun', 'winnertakeall', 'bomberger', 'texting', 'becasue', 
             'jpl', 'crete', 'richard', 'unrra', 'carlos', 'eckman', 'arturo', 'venice', 'da', 'blimplike', 'al', 'tabletennis', 'obama', 'egyptian', 
             'arent', 'nearrecord', 'malin', 'propername', 'atlantic', 'michael', 'elisabeth', 'thomas', 'selsky', 
              'carintensive', 'robert', 'mph', 'brin', 'plumer', 'shouldnt', 'heshe', 'sergey', 'nixon', "d'alto", 
              "vauban's", 'texas', 'european',
        ]
    
        self.add_prompts()
        self.add_features()
        # self.check()
    
    def add_prompts(self):
        prompts = [
            "Instruction: Evaluating the text and calculating content and wording score. Text: "
        ]
        self.train[self.feature_column] = prompts[0] + self.train[self.feature_column]
        self.test[self.feature_column] = prompts[0] + self.test[self.feature_column]
    
    # def count_stopwords(text, stopwords):
    #     '''Function that count a number of words which is not stopwords of nltk. '''
    #     text = text.split()
    #     stopwords_length = len([t for t in text if t in stopwords])
    #     return stopwords_length
    
    def add_features(self):
        df_concat = pd.concat([self.train, self.test])

        df_concat['letters'] = df_concat['full_text'].apply(lambda x: len(x))

        df_concat['words'] = df_concat['full_text'].apply(lambda x: len(x.split()))

        df_concat['unique_words'] = df_concat['full_text'].apply(lambda x: len(set(x.split())))

        df_concat['sentences'] = df_concat['full_text'].apply(lambda x: len(x.split('.')))

        df_concat['paragraph'] = df_concat['full_text'].apply(lambda x: len(x.split('\n\n')))


        # df['stopwords'] = df['full_text'].apply(count_stopwords, args=(stopwords,))

        # df['not_stopwords'] = df['words'] - df['stopwords']

        df_concat['spelling_errors'] = df_concat['full_text'].apply(self.count_spelling_errors)

        df_concat['cleaned_text'] = df_concat['full_text'].apply(self.dataPreprocessing)

        df_train = pd.read_csv(to_absolute_path("datasets/prediction_score.csv"))
        prediction_train_elements = df_train[['essay_id','prediction_score']] 


        # self.test.loc[:,'prediction_score'] = 3    
        # prediction_test_elements = self.test[['essay_id','prediction_score']]

        # prediction_elements = pd.concat([prediction_train_elements,prediction_test_elements],ignore_index=True)

        df_concat = df_concat.merge(prediction_train_elements, on='essay_id', how='left')
  
        df_concat.to_csv('df_concat.csv')

        vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            strip_accents='unicode',
            analyzer = 'word',
            ngram_range=(1,3),
            min_df=0.05,
            max_df=0.95,
            sublinear_tf=True,
        )

        train_tfid = vectorizer.fit_transform([i for i in df_concat['cleaned_text']])

        n_clusters = 7
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed).fit(train_tfid.toarray())
        labels = kmeans.predict(train_tfid.toarray())
        df_concat['group'] = labels



        # self.train = df_concat[:len(self.train)]
        # self.test = df_concat[len(self.train):]
        # self.test.drop(self.target_column, axis=1, inplace=True)
        self.train = df_concat.iloc[:len(self.train)].copy()
        self.test = df_concat.iloc[len(self.train):].copy()
        # test データから target_column を削除
        self.test = self.test.drop(self.target_column, axis=1)
        self.feature_columns = ['letters', 'words', 'unique_words', 'sentences', 'paragraph', 'group','prediction_score']
        self.feature_column = 'cleaned_text'
        # print(df_concat.columns)

    def removeHTML(self, x):
        html=re.compile(r'<.*?>')
        return html.sub(r'',x)

    def dataPreprocessing(self, x):
        # Convert words to lowercase
        x = x.lower()
        # Remove HTML
        x = self.removeHTML(x)
        # Delete strings starting with @
        x = re.sub("@\w+", '',x)
        # Delete Numbers
        x = re.sub("'\d+", '',x)
        x = re.sub("\d+", '',x)
        # Delete URL
        x = re.sub("http\w+", '',x)
        # Replace consecutive empty spaces with a single space character
        x = re.sub(r"\s+", " ", x)
        # Replace consecutive commas and periods with one comma and period character
        x = re.sub(r"\.+", ".", x)
        x = re.sub(r"\,+", ",", x)
        # Remove empty characters at the beginning and end
        x = x.strip()
        return x
    
    # full_textカラムに含まれる未定義単語の検出
    def find_unknown_words(self, text):
        # アポストロフィを除外した句読点のリストを作成
        punctuation_without_apostrophe = string.punctuation.replace("'", "")

        # 句読点を削除するための変換テーブルを作成
        trans_table = str.maketrans('', '', punctuation_without_apostrophe)

        # 変換テーブルを使用して句読点を削除
        text = text.translate(trans_table)
        words = text.split()
        return self.spell.unknown(words)
    
    # 未定義単語の確認の場合のみ実行する
    def check(self):
        # 全てのテキストから未定義単語を収集
        unknown_words_counter = Counter()
        for text in self.train['full_text']:
            unknown_words_counter.update(self.find_unknown_words(text))
        
        sorted_unknown_words = [word for word, count in unknown_words_counter.most_common(200)]

        # 未定義単語の表示（上位10個）
        print(sorted_unknown_words)
        exit()

    def count_spelling_errors(self, text):
        # アポストロフィを除外した句読点のリストを作成
        punctuation_without_apostrophe = string.punctuation.replace("'", "")

        # 句読点を削除するための変換テーブルを作成
        trans_table = str.maketrans('', '', punctuation_without_apostrophe)

        # 変換テーブルを使用して句読点を削除
        text = text.translate(trans_table)
        words = text.split()
        misspelled = self.spell.unknown(words)
        return len(misspelled)


