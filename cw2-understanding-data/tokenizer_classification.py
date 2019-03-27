# _*_ coding:utf-8 _*_
import os

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import nltk
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.probability import FreqDist
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.manifold import MDS

from scipy.cluster.hierarchy import ward, dendrogram

from gensim import corpora, models


# filter stopwords
stopwords = set(stopwords.words('english'))
en_words = set(words.words())
stemmer = SnowballStemmer('english')
book_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
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
          'THE HISTORY OF ROME. VOL. III',
          'THE HISTORY OF TACITUS. BOOK I. VOL. IV',
          'THE FLAVIUS JOSEPHU',
          ]


def tokenize_and_filter_punc(raw_text):
    tokens = wordpunct_tokenize(raw_text)
    text = nltk.Text(tokens)
    words = [w.lower() for w in text if w.isalpha()]

    return words

def tokenize_and_filter_stopwords(raw_text):
    tokens = wordpunct_tokenize(raw_text)
    text = nltk.Text(tokens)
    words = [w.lower() for w in text if w.isalpha()]
    words = [word for word in words if word not in stopwords]

    return words

def tokenize_sentences(raw_text):
    sents = sent_tokenize(raw_text)

    return sents


class BookObject(object):
    """
    每一本书籍建立一个类的对象
    """
    def __init__(self, title, contents, raw_texts=''):
        self.__title = title
        self.__contents = contents
        self.__raw_texts = raw_texts

    @property
    def title(self):
        return self.__title

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
    特征工程：计算文本单词数量，统计高频名词
    tf-idf方法
    doc2vec方法
    k-means方法
    hierachical方法
    mean-shift方法
    LDA方法
    """

    def __init__(self):
        self.__books = []
        self.__no_of_features = 0
        self.__total_vocabularies = []
        self.__tf_idf_matrix = None
        self.__d2v_model = None
        self.__d2v_vector = None

    @property
    def tf_idf_matrix(self):
        return self.__tf_idf_matrix

    @property
    def d2v_vector(self):
        return self.__d2v_vector

    def add_a_book(self, book):
        self.__books.append(book)

    # 特征工程
    # 打印单词数量
    def print_word_count(self):
        words_count = {}
        for book in self.__books:
            text_arr = book.raw_texts.strip().split()
            count = 0
            for w in text_arr:
                if len(w) > 0:
                    count += 1

            words_count[book.title] = count

        print('The word counts of each book is:\n')
        for key, value in words_count.items():
            print(key, ':', value)

    # 打印tokenize之后的单词数量
    def print_word_count_after_tokenization(self):
        words_count = {}
        for book in self.__books:
            text = book.raw_texts.strip()
            res = tokenize_and_filter_punc(text)
            words_count[book.title] = len(res)

        print('The word counts of each book after tokenize is:\n')
        for key, value in words_count.items():
            print(key, ':', value)

    # 输出每本书使用频率前20名的单词
    def highfreq_noun(self):
        print('High frequency noun in each book:\n')
        for book in self.__books:
            text = book.raw_texts
            tokens = tokenize_and_filter_punc(text)
            freq_dist = FreqDist(tokens)
            tagged = nltk.pos_tag(tokens)
            minfo = dict(freq_dist)
            info = list(set([k.lower() for k, v in tagged if v == 'NN']))
            kinfo = [(k, minfo.get(k)) for k in info]
            kinfo.sort(key=lambda k: k[1], reverse=True)
            kinfo = [w for w in kinfo if len(w[0]) > 3]
            output1 = ",".join([m[0] for m in kinfo[:20]])
            print('《', book.title[:-4], '》:\n', output1)
            output2 = " ".join([m[0] for m in kinfo[:20]])
            this_wordcloud = WordCloud().generate(output2)
            plt.imshow(this_wordcloud)
            plt.axis('off')
            plt.show()

    # 计算dist_matrix
    def dist_matrix(self, vector):
        dist_matrix = 1 - cosine_similarity(vector)

        return dist_matrix

        # TF-IDF计算词频

    def tf_idf(self):
        tfidf_matrix = None
        print('Training TF-IDF model...')
        tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000, min_df=0.01, stop_words='english',
                                           use_idf=True, tokenizer=tokenize_and_filter_punc, ngram_range=(1, 1))
        books = []
        for book in self.__books:
            books.append(book.raw_texts)

        tfidf_matrix = tfidf_vectorizer.fit_transform(books)

        self.__tf_idf_matrix = tfidf_matrix

        print('TF-IDF Done...')

    # Doc2vec
    def doc2vec(self, output_size=300, epoch=100):
        books = []
        for book in self.__books:
            books.append(book.raw_texts)

        tagged_docs = []
        # prepare doc2vec input - list of taggedDocument
        print('Start Tagging...')
        for index, text in enumerate(books):
            docTokens = tokenize_and_filter_stopwords(text)
            #             print("tagging:" + str(index))
            tagged_docs.append(models.doc2vec.TaggedDocument(docTokens, [str(index) + "_" + titles[index]]))
        print('Tagging Done')

        # setup configurations
        d2vm = models.Doc2Vec(vector_size=output_size, min_count=0, alpha=0.025, min_alpha=0.025)
        d2vm.build_vocab(tagged_docs)

        print("Training doc2vec model..")
        # Train the doc2vec model
        for epoch in range(epoch):  # number of epoch
            #             print("training ep:" + str(epoch))
            d2vm.train(tagged_docs, total_examples=len(tagged_docs), epochs=1)
            # Change learning rate for next epoch (start with large num to speed up at first and then decrease to fine grain learning)
            d2vm.alpha -= 0.002
            d2vm.min_alpha = d2vm.alpha
        # d2vm.train(tagged_docs, total_examples=len(tagged_docs), epochs=epoch )
        print("Done training..")
        ##d2vm.save('doc2vec.model')
        self.__d2v_model = d2vm

    # 转换为向量
    def doc2vec_to_vectors(self):
        # Extract vectors from doc2vec model
        feature_vectors = []
        for i in range(0, len(self.__d2v_model.docvecs)):
            feature_vectors.append(self.__d2v_model.docvecs[i])

        self.__d2v_vector = feature_vectors

    # K-means算法
    def k_mean_clustering(self, matrix, dist_matrix, plot_title, n_clusters=5):
        num_clusters = n_clusters

        print('Start K-Means Clustering...')
        km = KMeans(n_clusters=num_clusters)
        km.fit(matrix)
        clusters = km.labels_.tolist()
        print(clusters)

        synopses = []
        for book in self.__books:
            synopses.append(book.raw_texts)

        books = {'book_id': book_id, 'title': titles, 'synopsis': synopses, 'cluster': clusters}

        frame = pd.DataFrame(books, columns=['book_id', 'title', 'cluster'])
        print(frame)

        # convert two components as we're plotting points in a two-dimensional plane
        # "precomputed" because we provide a distance matrix
        # we will also specify `random_state` so the plot is reproducible.
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

        pos = mds.fit_transform(dist_matrix)  # shape (n_components, n_samples)
        xs, ys = pos[:, 0], pos[:, 1]

        # set up colors per clusters using a dict
        cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

        # set up cluster names using a dict
        cluster_names = {0: 'Group 1',
                         1: 'Group 2',
                         2: 'Group 3',
                         3: 'Group 4',
                         4: 'Group 5'}

        # create data frame that has the result of the MDS plus the cluster numbers and titles
        df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, book_id=book_id))

        # group by cluster
        groups = df.groupby('label')

        # set up plot
        fig, ax = plt.subplots(figsize=(15, 9))  # set size
        ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

        # iterate through groups to layer the plot
        # note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                    label=cluster_names[name], color=cluster_colors[name],
                    mec='none')
            ax.set_aspect('auto')
            ax.tick_params( \
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)
            ax.tick_params( \
                axis='y',  # changes apply to the y-axis
                which='both',  # both major and minor ticks are affected
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelleft=False)

        ax.legend(numpoints=1)  # show legend with only 1 point
        ax.set_title(plot_title, fontsize=24)

        # add label in x,y position with the label as the film title
        for i in range(len(df)):
            ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['book_id'], fontsize=18)

        plt.show()

    # 层级聚类
    def hierachical_clustering(self, dist_matrix, plot_title):
        linkage_matrix = ward(dist_matrix)  # define the linkage_matrix using ward clustering pre-computed distances

        fig, ax = plt.subplots(figsize=(15, 9))  # set size
        ax = dendrogram(linkage_matrix, orientation="right", labels=book_id)

        plt.tick_params( \
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelsize=20)
        plt.title(plot_title, fontsize=24)

        fig.set_tight_layout(True)
        plt.show()

    # mean_shift
    def mean_shift(self, matrix, dist_matrix, plot_title):
        print('Mean Shift Training...')
        bandwidth = estimate_bandwidth(matrix, quantile=0.2, n_samples=500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(matrix)
        print('Mean Shift Done...')
        labels = ms.labels_

        books = {'book_id': book_id, 'title': titles, 'labels': labels}

        frame = pd.DataFrame(books, columns=['book_id', 'title', 'labels'])
        print(frame)

        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
        pos = mds.fit_transform(dist_matrix)
        xs, ys = pos[:, 0], pos[:, 1]

        df = pd.DataFrame(dict(x=xs, y=ys, label=labels, book_id=book_id))
        groups = df.groupby('label')

        # set up plot
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.margins(0.05)

        cluster_names = {0: 'Group 1',
                         1: 'Group 2',
                         2: 'Group 3',
                         3: 'Group 4',
                         4: 'Group 5',
                         5: 'Group 6'}

        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', label=cluster_names[name], ms=12, mec='none')
            ax.set_aspect('auto')
            ax.tick_params( \
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)
            ax.tick_params( \
                axis='y',  # changes apply to the y-axis
                which='both',  # both major and minor ticks are affected
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelleft=False)

        ax.legend(numpoints=1)  # show legend with only 1 point
        ax.set_title(plot_title, fontsize=24)

        # add label in x,y position with the label as the film title
        for i in range(len(df)):
            ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['book_id'], fontsize=18)

        plt.show()

    # LDA
    def LDA(self):
        books_words = []
        for book in self.__books:
            raw_texts = book.raw_texts
            words_ls = tokenize_and_filter_stopwords(raw_texts)
            books_words.append(words_ls)

        print('Start LDA...')
        # 创建一个Gensim词典
        dictionary = corpora.Dictionary(books_words)
        dictionary.filter_extremes(no_below=1, no_above=0.8)
        # 将词典转换为一个Bag of Words
        corpus = [dictionary.doc2bow(book_words) for book_words in books_words]

        lda = models.LdaModel(corpus, num_topics=3,
                              id2word=dictionary,
                              update_every=5,
                              chunksize=10000,
                              passes=100)

        print('LDA Done...')
        topics = lda.print_topics(num_topics=3, num_words=20)
        # print(topics_matrix)
        #         topics_matrix = np.array(topics_matrix, dtype=object)
        return topics

    # 加载原始文本并运行
    def load_raw_text(self):
        rootdir = '/Users/suhang/Documents/GitHub/COMP6237-Data-Mining/cw2-understanding-data/raw_text'
        files = os.listdir(rootdir)
        for file in files:
            if file != '.DS_Store':
                file_path = os.path.join(rootdir, file)
                if os.path.isfile(file_path):
                    with open(file_path) as f:
                        raw_texts = f.read()

                        book = BookObject(file, {}, raw_texts=raw_texts)
                        self.add_a_book(book)




if __name__ == '__main__':
    # 创建实例，加载文本
    test = TextClassification()
    test.load_raw_text()

    # 探索数据
    # test.print_word_count()
    # test.print_word_count_after_tokenization()

    # 用TF-IDF获得向量进行聚类
    print("TF-IDF:\n")
    test.tf_idf()
    tf_idf_matrix = test.tf_idf_matrix
    tf_idf_distm = test.dist_matrix(tf_idf_matrix)
    test.k_mean_clustering(tf_idf_matrix, tf_idf_distm, 'K-Means Clustering for TF-IDF')
    test.hierachical_clustering(tf_idf_distm, 'Hierachical Clustering for TF-IDF')
    # test.mean_shift(tf_idf_matrix)
    print('\n')

    # 用Doc2vec获得向量进行聚类
    print("Doc2Vec:\n")
    test.doc2vec()
    test.doc2vec_to_vectors()
    d2v_vector = test.d2v_vector
    d2v_distm = test.dist_matrix(d2v_vector)
    test.k_mean_clustering(d2v_vector, d2v_distm, 'K-Means Clustering for Doc2vec')
    test.hierachical_clustering(d2v_distm, 'K-Means Clustering for Doc2vec')
    test.mean_shift(d2v_vector, d2v_distm, 'Mean Shift Clustering for Doc2vec')
    print('\n')

    # LDA
    print('LDA topics:\n')
    topics = test.LDA()
    for topic in topics:
        print(topic)

