import numpy as np
import gensim 
import cloudpickle
from sys import version_info
import mlflow

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
        #import gensim 
        #print(context)
        #self.model = gensim.models.KeyedVectors.load(context.artifacts["gensim_model"], mmap='r').wv
        
    def predict(self, model, data):
        return self.word2vec(data)

    def word2vec(self, data):
        corpus = data["description_preprocessed"].apply(lambda elem: list(elem)).to_list()  

        '''Get Word2Vec embeddings.   
        Input:    
            corpus: list of preprocessed strings.   
        Output:    
            embeddings: array of shape [n_sample, dim]    
        '''
        import numpy as np  
        embeddings = [] 
        # drop tokens which not in vocab
        for tokens in corpus:
            tokens = [token for token in tokens if token in self.model.wv.vocab]
            #logger.info(', '.join(tokens))
            print(tokens)
            if len(tokens)==0:
                embedding = np.zeros(100).tolist()
            else:
                embedding = np.mean(self.model[tokens],axis=0).tolist()
            embeddings.append(embedding)
            print(embedding)
        embeddings = np.array(embeddings)
        return embeddings
