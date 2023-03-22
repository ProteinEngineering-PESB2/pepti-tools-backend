from bio_embeddings.embed import BeplerEmbedder ##
from bio_embeddings.embed import CPCProtEmbedder
from bio_embeddings.embed import ESM1bEmbedder ##
from bio_embeddings.embed import ESMEmbedder ##
from bio_embeddings.embed import ESM1vEmbedder
from bio_embeddings.embed import FastTextEmbedder ##
from bio_embeddings.embed import GloveEmbedder ##
from bio_embeddings.embed import OneHotEncodingEmbedder ##
from bio_embeddings.embed import PLUSRNNEmbedder ##
from bio_embeddings.embed import ProtTransAlbertBFDEmbedder ##
from bio_embeddings.embed import ProtTransBertBFDEmbedder
from bio_embeddings.embed import ProtTransT5BFDEmbedder
from bio_embeddings.embed import ProtTransT5UniRef50Embedder
from bio_embeddings.embed import ProtTransT5XLU50Embedder
from bio_embeddings.embed import ProtTransXLNetUniRef100Embedder
from bio_embeddings.embed import SeqVecEmbedder ##
from bio_embeddings.embed import UniRepEmbedder ##
from bio_embeddings.embed import Word2VecEmbedder ##
import pandas as pd
import numpy as np
from tqdm import tqdm

class UsingBioembeddings(object):

    def __init__(
            self,
            dataset=None,
            column_id=None,
            column_seq=None,
            is_reduced=True,
            device = None
            ):
        
        self.dataset = dataset
        self.column_id = column_id
        self.column_seq = column_seq
        self.is_reduced=is_reduced
        self.device = device

        # to save the results
        self.embedder = None
        self.embeddings = None
        self.np_data = None

    def __apply_embedding(self, model, ensemble_id = None, half_precision_model = None):
        if self.device != None:
            self.embedder = model(device=self.device)
        else:
            self.embedder = model()
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()
        return self.parse_output()

    def __reducing(self):
        self.np_data = np.zeros(shape=(len(self.dataset), self.embedder.embedding_dimension))
        for idx, embed in tqdm(enumerate(self.embeddings), desc="Reducing embeddings"):
            self.np_data[idx] = self.embedder.reduce_per_protein(embed)

    def apply_bepler(self):
        return self.__apply_embedding(BeplerEmbedder)

    def apply_cpcprot(self):
        return self.__apply_embedding(CPCProtEmbedder)
    
    def apply_esm1b(self):
        return self.__apply_embedding(ESM1bEmbedder)

    def apply_esme(self):
        return self.__apply_embedding(ESMEmbedder)


    def apply_fasttext(self):
        return self.__apply_embedding(FastTextEmbedder)

    def apply_glove(self):
        return self.__apply_embedding(GloveEmbedder)

    def apply_onehot(self):
        return self.__apply_embedding(OneHotEncodingEmbedder)

    def apply_plus_rnn(self):
        return self.__apply_embedding(PLUSRNNEmbedder)

    def apply_prottrans_albert(self):
        return self.__apply_embedding(ProtTransAlbertBFDEmbedder)

    def apply_prottrans_bert(self):
        return self.__apply_embedding(ProtTransBertBFDEmbedder)

    def apply_prottrans_T5BFD(self):
        return self.__apply_embedding(ProtTransT5BFDEmbedder)

    def apply_prottrans_T5_UniRef(self):
        return self.__apply_embedding(ProtTransT5UniRef50Embedder)

    def apply_prottrans_XLNetUniRef(self):
        return self.__apply_embedding(ProtTransXLNetUniRef100Embedder)

    def apply_seqvec(self):
        return self.__apply_embedding(SeqVecEmbedder)
    
    def apply_unirep(self):
        return self.__apply_embedding(UniRepEmbedder)

    def apply_word2vec(self):
        return self.__apply_embedding(Word2VecEmbedder)
    
    def parse_output(self):
        header = ["p_{}".format(i) for i in range(len(self.np_data[0]))]
        df_data_encode = pd.DataFrame(self.np_data, columns=header)
        df_data_encode[self.column_id] = self.dataset[self.column_id]
        df_data_encode = df_data_encode[[self.column_id] + header]
        return df_data_encode
    
    def apply_prottrans_T5_XLU50(self):
        if self.device != None:
            self.embedder = ProtTransT5XLU50Embedder(half_precision_model=True, device=self.device)
        else:
            self.embedder = ProtTransT5XLU50Embedder(half_precision_model=True)
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()
        return self.parse_output()
    
    def apply_esm1v(self, ensemble_id=5):
        if self.device != None:
            self.embedder = ESM1vEmbedder(ensemble_id=ensemble_id, device=self.device)
        else:
            self.embedder = ESM1vEmbedder(ensemble_id=ensemble_id)
        
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        
        if self.is_reduced == True:
            self.__reducing()
        return self.parse_output()