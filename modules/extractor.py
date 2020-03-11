from model.albert import AlbertModel
from model.tokenizer import AlbertTokenizer
from tools.configreader import model_config,absolute_path
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Extractor():
    def __init__(self):
        self.model_path = absolute_path(model_config.get('Path','model'))
        self.config_path = absolute_path(model_config.get('Path','config'))
        self.vocab_path = absolute_path(model_config.get('Path','vocab'))
        self.model,self.tokenizer = self.load_model()

    def load_model(self):
        model = AlbertModel.from_pretrained(self.model_path,config=self.config_path)
        tokenizer = AlbertTokenizer(self.vocab_path)
        return model,tokenizer

    def feature_extractor(self,corpus):
        tokens = [self.tokenizer.encode(i) for i in corpus]
        all_vec = []
        for token in tqdm(tokens):
            with torch.no_grad():
                output = self.model(torch.Tensor(token).long().unsqueeze(0).to(device))
            vec = output[0][:,0,:]
            all_vec.append(vec)

        vecs = torch.cat(all_vec,dim=0)
        return vecs
