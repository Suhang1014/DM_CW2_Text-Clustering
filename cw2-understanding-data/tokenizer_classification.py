# _*_ coding:utf-8 _*_
import os

import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

from scipy.cluster.hierarchy import ward, dendrogram

# filter stopwords
stopwords = set(stopwords.words('english'))
en_words = set(words.words())
stemmer = SnowballStemmer('english')
titles = ['THE HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE. VOL. VI',
          'THE HISTORIES CAIUS COBNELIUS TACITUS',
          'THE WORK OF JOSEPH US, THE JEWISH WAR. VOL. IV',
          'THE HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE. VOL, I',
          'THE HISTORY OF TACITUS. BOOK I. VOL. V',
          'THE FIRST AND THIRTY-THIRD BOOKS OF PLINY\'S NATURAL HISTORY',
          'THE HISTORY OF THE ROMAN EMPIRE. VOL. V',
          'THE HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE. VOL. II',
          'THE HISTORY OF THE PELOPONNESIAN WAR. VOL. II',
          'TITUS LIVIUS\' ROMAN',
          'THE HISTORY OF ROME,  BY TITUS LIVIUS. VOL. I',
          'THE HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE. VOL. IV',
          'DICTIONARY GREEK AND ROMAN GEOGRAPHY. VOL. II',
          'THE LEARNED AND AUTHENTIC JEWISH HISTORIAN AND CELEBRATED WARRIOR. VOL. III',
          'LIVY. VOL. III',
          'LIVY. VOL. V',
          'THE HISTORICAL ANNALS OF CORNELIUS TACITUS. VOL. I',
          'THE HISTORY OF THE PELOPONNESIAN WAR. VOL. I',
          'THE LEARNED AND AUTHENTIC JEWISH HISTORIAN,  AND CELEBRATED WARRIOR. VOL. IV',
          'THE DESCRIPTION OF GREECE',
          'THE HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE. VOL. III',
          'TTHE HISTORY OF ROME. VOL. III',
          'HE HISTORY OF TACITUS. BOOK I. VOL. IV',
          'THE FLAVIUS JOSEPHU',
          ]


def tokenize_and_filter_punc(raw_text):
    tokens = nltk.wordpunct_tokenize(raw_text)
    text = nltk.Text(tokens)
    words = [w.lower() for w in text if w.isalpha()]

    return words


class BookObject(object):
    """
    每一本书籍建立一个类的对象
    """
    def __init__(self, title, contents, raw_texts=''):
        self.__title = title
        self.__contents = contents
        self.__raw_texts = raw_texts

    @property
    def raw_texts(self):
        return self.__raw_texts

    @property
    def contents(self):
        return self.__contents.keys()


class TextClassification(object):
    """
    添加方法：
    添加一本书
    加载原始文本
    tf-idf方法
    k-means方法
    hierachical方法
    """
    def __init__(self):
        self.__books = []
        self.__tf_idf_matrix = None
        self.__dist_matrix = None

    def add_a_book(self, book):
        self.__books.append(book)

    def load_raw_text(self):
        rootdir = '/Users/suhang/Documents/GitHub/COMP6237-Data-Mining/cw2-understanding-data/raw_text'
        files = os.listdir(rootdir)
        for file in files:
            print(file)
            if file != '.DS_Store':
                file_path = os.path.join(rootdir, file)
                if os.path.isfile(file_path):
                    with open(file_path) as f:
                        raw_texts = f.read()

                        book = BookObject(f, {}, raw_texts=raw_texts)
                        self.add_a_book(book)

    def tf_idf(self):
        tfidf_matrix = None
        tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000, min_df=0.01, stop_words='english',
                                           use_idf=True,  tokenizer=tokenize_and_filter_punc, ngram_range=(1, 1))
        books = []
        for book in self.__books:
            books.append(book.raw_texts)

        tfidf_matrix = tfidf_vectorizer.fit_transform(books)

        self.__tf_idf_matrix = tfidf_matrix
        print(self.__tf_idf_matrix.shape)

        self.__dist_matrix = 1 - cosine_similarity(self.__tf_idf_matrix)
        print(self.__dist_matrix)

    def k_mean_clustering(self, n_clusters=5):
        num_clusters = n_clusters

        km = KMeans(n_clusters=num_clusters)
        km.fit(self.__tf_idf_matrix)
        clusters = km.labels_.tolist()
        print(clusters)


        synopses = []
        for book in self.__books:
            synopses.append(book.raw_texts)

        books = {'title': titles, 'synopsis': synopses, 'cluster': clusters}

        frame = pd.DataFrame(books, index = [clusters] , columns = ['title', 'cluster'])
        print(frame['cluster'].value_counts())
        #grouped = frame['title'].groupby(frame['cluster']) #groupby cluster for aggregation purposes

        #print grouped.mean() #average rank (1 to 100) per cluster

        # convert two components as we're plotting points in a two-dimensional plane
        # "precomputed" because we provide a distance matrix
        # we will also specify `random_state` so the plot is reproducible.
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

        pos = mds.fit_transform(self.__dist_matrix)  # shape (n_components, n_samples)
        xs, ys = pos[:, 0], pos[:, 1]
        print(xs)
        print(ys)

        #set up colors per clusters using a dict
        cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

        #set up cluster names using a dict
        cluster_names = {0: 'Group 1',
                         1: 'Group 2',
                         2: 'Group 3',
                         3: 'Group 4',
                         4: 'Group 5'}

        #create data frame that has the result of the MDS plus the cluster numbers and titles
        df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))

        #group by cluster
        groups = df.groupby('label')


        # set up plot
        fig, ax = plt.subplots(figsize=(17, 9)) # set size
        ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

        #iterate through groups to layer the plot
        #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                    label=cluster_names[name], color=cluster_colors[name],
                    mec='none')
            ax.set_aspect('auto')
            ax.tick_params(\
                axis= 'x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
            ax.tick_params(\
                axis= 'y',         # changes apply to the y-axis
                which='both',      # both major and minor ticks are affected
                left='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelleft='off')

        ax.legend(numpoints=1)  #show legend with only 1 point

        #add label in x,y position with the label as the film title
        for i in range(len(df)):
            ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

        plt.show() #show the plot

    def hierachical_clustering(self):
        linkage_matrix = ward(self.__dist_matrix) #define the linkage_matrix using ward clustering pre-computed distances

        fig, ax = plt.subplots(figsize=(15, 9)) # set size
        ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

        plt.tick_params(\
            axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')

        fig.set_tight_layout(True) #show plot with tight layout
        plt.show()



if __name__ == '__main__':
    test = TextClassification()
    test.load_raw_text()
    test.tf_idf()
    test.k_mean_clustering()
    test.hierachical_clustering()

