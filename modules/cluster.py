import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import torch
from modules.extractor import Extractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Cluster():
    def __init__(self):
        self.extractor = Extractor()

    def euclidean(self,A):
        mm_product = torch.mm(A, A.t())
        diag = torch.diag(mm_product)
        sqed = diag.unsqueeze(0) - 2 * mm_product + diag.unsqueeze(1)
        sqed = sqed.cpu().numpy()
        sqed = sqed[~np.greater_equal.outer(np.arange(len(sqed)), np.arange(len(sqed)))]
        sqed[sqed < 0] = 0.0
        return sqed

    def cosine(self,A):
        mm_product = torch.mm(A, A.t())
        dot_product = torch.mul(A, A)
        mode = dot_product.sum(dim=1, keepdim=True).sqrt()
        mm_mode = torch.mm(mode, mode.t()) + 1e-10
        cos = torch.ones(mm_product.shape)-mm_product / mm_mode
        cos = cos.cpu().numpy()
        sqed = cos[~np.greater_equal.outer(np.arange(len(cos)), np.arange(len(cos)))]
        return sqed

    def sch_cluster(self,distance,threshold=0.3):
        z = sch.linkage(distance, method='complete')
        labels = sch.fcluster(z, t=threshold, criterion='distance')
        return labels

    def cluster(self,corpus,threshold=0.1,use='sklearn'):
        vecs = self.extractor.feature_extractor(corpus)
        vecs = vecs.to(device)

        if use=='scipy':
            distance_mat = self.cosine(vecs)
            labels = self.sch_cluster(distance_mat,threshold)
        else:
            agg = AgglomerativeClustering(distance_threshold=threshold,n_clusters=None,
                                          linkage='complete',affinity='cosine')
            agg.fit(vecs.cpu().numpy())
            labels = agg.labels_.tolist()
        return labels

    def file_reader(self,file_path):
        with open(file_path,'r',encoding='utf8') as f:
            lines = f.readlines()
        lines = [i.strip() for i in lines]
        return lines

    def process(self,file_path,threshold=0.04,use='scipy'):
        # scipy label从1开始，sklearn label从0开始
        corpus = self.file_reader(file_path)
        corpus = list(set(corpus))
        labels = self.cluster(corpus,threshold,use=use)
        res = list(zip(corpus,labels))
        res.sort(key=lambda x:x[1])
        return res


if __name__ == '__main__':
    cluster = Cluster()
    res = cluster.process('../data/test.txt')
    print(res)










