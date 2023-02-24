import numpy as np
import gensim 
import cloudpickle


PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)

CONDA_ENV = {
    'channels': ['defaults'],
    'dependencies': [
      'python={}'.format(PYTHON_VERSION),
      'pip',
      {
        'pip': [
          'mlflow',
          'gensim=={}'.format(gensim.__version__),
          'cloudpickle=={}'.format(cloudpickle.__version__),
        ],
      },
    ],
    'name': 'gensim_env'
}

class GensimModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def load_context(self, context):
        '''Load embedding model including word2vec.
        Input: 
            context: string. Path of model.
        Output:
            model: model object.
        '''
        #self.model = Word2Vec.load(context.artifacts["gensim_model"])
        self.model = KeyedVectors.load(context.artifacts["gensim_model"], mmap='r').wv
        
    def predict(self, model, data):
        return self.word2vec(data)

    def word2vec(self, corpus):
        '''Get Word2Vec embeddings.   
        Input:    
            corpus: list of preprocessed strings.   
        Output:    
            embeddings: array of shape [n_sample, dim]    
        '''
        embeddings = [] 
        # drop tokens which not in vocab
        for text in corpus:
            print(text)
            tokens = text.split(' ')
            tokens = [token for token in tokens if token in self.model.vocab]
            #logger.info(', '.join(tokens))
            print(tokens)
            if len(tokens)==0:
                embedding = self.model['unk'].tolist()
            else:
                embedding = np.mean(self.model[tokens],axis=0).tolist()
            embeddings.append(embedding)
            print(embedding)
        embeddings = np.array(embeddings)
        return embeddings
