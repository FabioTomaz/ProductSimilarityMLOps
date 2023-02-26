# coding: utf-8
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import logging 
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s" 
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT) 
logger = logging.getLogger(__file__)



class Embeddings():
    def __init__(self):
        self.model = None

    def load_model(self, method='word2vec', model_path=''):
        '''Load embedding model including word2vec/fasttext/glove/bert.
        Input: 
            method: string. Including "word2vec"/"fasttext"/"glove"/"bert".
            model_path: string. Path of model.
        Output:
            model: model object.
        '''
        self.method = method
        self.model_path = model_path
        if model_path == '':
            self.model = None
            return None
        logger.info('Load embedding model...')
        if method in ['word2vec','glove']:
            from gensim.models import KeyedVectors
            if model_path[-4:]=='.txt':
                self.model = KeyedVectors.load_word2vec_format(model_path,binary=False).wv
            elif model_path[-4:] =='.bin':
                self.model = KeyedVectors.load_word2vec_format(model_path,binary=True).wv
            else:
                self.model = KeyedVectors.load(model_path,mmap='r').wv
            return self.model
        elif method == 'fasttext':
            from gensim.models.wrappers import FastText
            self.model = FastText.load_fasttext_format(model_path).wv
            return self.model
        elif method == 'bert':
            from transformers import BertTokenizer, BertModel 
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertModel.from_pretrained(model_path)
            return self.model
        else:
            self.model = None
            return None

    def bow(self, corpus, ngram_range=(1,1), min_df=1):
        '''Get BOW (bag of words) embeddings.
        Input:
            corpus: list of preprocessed strings. 
            ngram_range: tuple. (min_ngram, max_ngram) means min_ngram<=ngram<=max_ngram
            min_df: int. Mininum frequencey of a word. 
        Output:
            embeddings: array of shape [n_sample, dim]
        '''
        vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df, token_pattern='(?u)\\b\\w+\\b') 
        # The default token_pattern r'\b\w+\b' tokenizes the string by extracting words of at least 2 letters, which is not suitable for Chinese
        X = vectorizer.fit_transform(corpus)
        #print(vectorizer.get_feature_names())
        embeddings = X.toarray()
        return embeddings
    
    def tfidf(self, corpus, ngram_range=(1,1), min_df=1):
        '''Get TFIDF embeddings. 
        Input: 
            corpus: list of preprocessed strings.  
            ngram_range: tuple. (min_ngram, max_ngram) means min_ngram<=ngram<=max_ngram 
            min_df: int. Mininum frequencey of a word.  
        Output: 
            embeddings: array of shape [n_sample, dim] 
        '''
        transformer = TfidfTransformer(use_idf=True, smooth_idf=True)
        counts = self.bow(corpus, ngram_range, min_df)
        X = transformer.fit_transform(counts)
        embeddings = X.toarray()
        return embeddings
    
    def word2vec(self, corpus, method='word2vec', model_path=''):
        '''Get Word2Vec embeddings.   
        Input:    
            corpus: list of preprocessed strings.   
            method: string. "word2vec"/"glove"/"fasttext"
            model_path: string. Path of model.   
        Output:    
            embeddings: array of shape [n_sample, dim]    
        '''
        # load model
        if self.model is None and model_path!='':
            self.load_model(method, model_path)
        embeddings = [] 
        # drop tokens which not in vocab
        for text in corpus:
            tokens = text.split(' ')
            tokens = [token for token in tokens if token in self.model.vocab]
            #logger.info(', '.join(tokens))
            if len(tokens)==0:
                embedding = self.model['unk'].tolist()
            else:
                embedding = np.mean(self.model[tokens],axis=0).tolist()
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        return embeddings
   
    def bert(self, corpus, model_path='', mode='cls'):
        '''Get BERT embeddings.   
        Input:    
            corpus: list of preprocessed strings.   
            model_path: string. Path of model.
            mode: string. "cls"/"mean". "cls" mode: get the embedding of the first 
            token of a sentence; "mean" mode: get the average embedding of all tokens of 
            a sentence except for the first [CLS] and the last [SEP] tokens.   
        Output:    
            embeddings: array of shape [n_sample, dim]    
        '''
        import torch
        # load model
        if self.model is None and model_path!='':
            self.load_model('bert',model_path)
            
        embeddings = []
        for text in corpus:
            # tokenize and encode
            input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0) # Batch size 1
            # get embedding
            outputs = self.model(input_ids)
            embedding = outputs[0].detach().numpy()  # The last hidden-state is the first element of the output tuple
            if mode=='cls':
                embedding = embedding[0][0]
            elif mode=='mean':
                embedding = np.mean(embedding[0],axis=0)
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        return embeddings

           
    def get_embeddings(self, corpus, ngram_range=(1,1), min_df=1, dim=5, method='tfidf', model_path=''):
        '''Get embeddings according to params.
        Input:
            corpus: list of preprocessed strings.     
            ngram_range: tuple. (min_ngram, max_ngram) means min_ngram<=ngram<=max_ngram   
            min_df: int. Mininum frequencey of a word. 
            dim: int. Dimention of embedding.  
            method: string. Including "bow"/"tfidf"/"word2vec"/"glove"/"fasttext"/"bert"
        Output:
            embeddings: array of shape [n_sample, dim] 
        '''
        self.method = method 
        if self.method == 'bow': 
            return self.bow(corpus, ngram_range=ngram_range, min_df=min_df) 
        elif self.method == 'tfidf': 
            return self.tfidf(corpus, ngram_range=ngram_range, min_df=min_df) 
        elif self.method in ['word2vec','glove','fasttext']: 
            return self.word2vec(corpus, method=method, model_path=model_path)
        elif self.method == 'bert':
            return self.bert(corpus, model_path=model_path)


if __name__ == '__main__': 
    corpus_en = ['This is the first document.',
         'This is the second second document.',
         'And the third one!',
         'Is this the first document?']

    # text preprocess
    from preprocess import Preprocess
    tp_en = Preprocess('en')
    corpus_en = tp_en.preprocess(corpus_en)
    print(corpus_en)

    '''
    # bow
    emb = Embeddings()
    bow_en = emb.bow(corpus_en)
    print(bow_en)
    bow_en = emb.bow(corpus_en)
    print(bow_en)

    # tfidf
    emb = Embeddings() 
    tfidf_en = emb.tfidf(corpus_en)
    print(tfidf_en)
    tfidf_en = emb.tfidf(corpus_en)
    print(tfidf_en)
    
    # word2vec
    emb = Embeddings() 
    emb.load_model(method='word2vec',model_path="../w2v_models/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding") 
    word2vec_en = emb.word2vec(corpus_en) 
    print(word2vec_en)
    
    # glove
    # transform glove model to word2vec format
    from utils import transformGlove
    source_model_path = os.path.join(os.path.dirname(__file__),"../w2v_models/GloVe/en/glove.42B.300d.txt")
    target_model = os.path.join(os.path.dirname(__file__),"../w2v_models/GloVe/en/glove.42B.300d.w2v")
    #transformGlove(source_model_path,target_model,binary=True)
    target_model_path = target_model+'.bin'
    # get embeddings
    emb = Embeddings()
    emb.load_model(method='word2vec',model_path=target_model_path)
    glove_en = emb.word2vec(corpus_en)
    print(glove_en)
    
    # fasttext
    model_path = os.path.join(os.path.dirname(__file__),"../w2v_models/FastText/en/cc.en.300.bin")
    
    emb = Embeddings()
    emb.load_model(method='fasttext', model_path=model_path)
    ft_en = emb.word2vec(corpus_en)
    print(ft_en.shape)
    
    emb = Embeddings()
    model_path = os.path.join(os.path.dirname(__file__),"../../bert-base-chinese")
    emb.load_model(method='bert', model_path=model_path)
    bert_en = emb.bert(corpus_en)
    print(bert_en.shape)
    '''
    emb = Embeddings() 
    bow_en = emb.bow(corpus_en) 
    print(bow_en) 
    bow_en = emb.bow(corpus_en) 
    print(bow_en)
