import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import utils.cca_core as cca_core
import utils.entrz_symbol as entrz_symbol
import pandas as pd
import pickle

# edit path here
RES_SAVE_PATH = "/PRISME/result"
BASE_SAVE_PATH = "/PRISME/dataset"
dat_names = ['gene2vec', 'omics', 'geneformer', 'prottrans', 'genept', 'biolinkbert_summary', 'struct2vec', 'k2bTransE', 'k2bMurE']

fit_svcca = True
PERMUTE = True
n_sample = 100

def process_dat(dat_i, dat_j, dat_names):
    dat1_name = dat_names[dat_i]
    dat2_name = dat_names[dat_j]
    dat1, dat1_gene = read_dat(dat1_name)
    dat2, dat2_gene = read_dat(dat2_name)

    gene_inter = set(dat1_gene).intersection(set(dat2_gene))

    # select the common genes
    indices = [i for i, gene in enumerate(dat1_gene) if gene in gene_inter]
    print("Indices type:", type(indices))
    print("Indices dtype before conversion:", np.array(indices).dtype)
    print("Sample indices:", indices[:5])
    dat1 = dat1[np.array([i for i,gene in enumerate(dat1_gene) if gene in gene_inter]), :]
    dat2 = dat2[np.array([i for i,gene in enumerate(dat2_gene) if gene in gene_inter]), :]

    # sort the data based on genes
    dat1_gene = [gene for gene in dat1_gene if gene in gene_inter]
    dat2_gene = [gene for gene in dat2_gene if gene in gene_inter]

    dat1 = dat1[np.argsort(dat1_gene), :]
    dat2 = dat2[np.argsort(dat2_gene), :]
    # dat1_gene = dat1_gene[np.argsort(dat1_gene)]
    # dat2_gene = dat2_gene[np.argsort(dat2_gene)]

    # standardize the data
    dat1 = (dat1 - np.mean(dat1, axis=0)) / np.std(dat1, axis=0)
    dat2 = (dat2 - np.mean(dat2, axis=0)) / np.std(dat2, axis=0)
    
    return dat1, dat2, dat1_name, dat2_name


def read_dat(data_type, n_dim=512):
    if data_type == 'gene2vec':
        if os.path.exists(f'{BASE_SAVE_PATH}/gene2vec.npy'):
            gene2vec = np.load(f'{BASE_SAVE_PATH}/gene2vec.npy')
            gene2vec_gene = np.load(f'{BASE_SAVE_PATH}/gene2vec_gene.npy', allow_pickle=True)
            return gene2vec, gene2vec_gene
        
        gene2vec = '/PRISME/dataset/gene2vec.txt'
        gene2vec = pd.read_csv(gene2vec, sep=' ', header=None, skiprows=1, index_col=0)
        gene2vec_gene = gene2vec.index.values
        gene2vec = np.array(gene2vec) #(n_gene, dim)
        gene2vec = gene2vec[:, :-1] 
        np.save(f'{BASE_SAVE_PATH}/gene2vec.npy', gene2vec)
        np.save(f'{BASE_SAVE_PATH}/gene2vec_gene.npy', gene2vec_gene)
    
    if data_type == 'omics':
        if os.path.exists(f'{BASE_SAVE_PATH}/omics.npy'):
            dat = np.load(f'{BASE_SAVE_PATH}/omics.npy')
            dat_gene = np.load(f'{BASE_SAVE_PATH}/omics_gene.npy')
            return dat, dat_gene
        
        dat = '/PRISME/dataset/omics.tsv'
        dat = pd.read_csv(dat, sep='\t')
        dat_gene = dat['gene_id'].values
        ensembl2symbol = {}
        # batchly convert ensembl ids to gene symbols with batch size = 50
        for i in tqdm(range(0, len(dat_gene), 50)):
            ensembl2symbol.update(entrz_symbol.batch_convert_ensembl_to_symbols(list(dat_gene[i:i+50])))

        dat_gene = [ensembl2symbol[gene] for gene in dat_gene]
        dat = np.array(dat.iloc[:, 1:])

        np.save(f'{BASE_SAVE_PATH}/{data_type}.npy', dat)
        np.save(f'{BASE_SAVE_PATH}/{data_type}_gene.npy', dat_gene)

    if data_type == 'geneformer':
        if os.path.exists(f'{BASE_SAVE_PATH}/geneformer.npy'):
            geneformer = np.load(f'{BASE_SAVE_PATH}/geneformer.npy')
            geneformer_gene = np.load(f'{BASE_SAVE_PATH}/geneformer_gene.npy')
            return geneformer, geneformer_gene
        
        geneformer = '/PRISME/dataset/gene_emb/code/Geneformer/gene_embedding.pkl'
        with open(geneformer,'rb') as f:
            geneformer = pickle.load(f)
        geneformer_gene = list(geneformer.keys())
        geneformer = np.stack([np.array(geneformer[k]) for k in geneformer], axis=0)
        np.save(f'{BASE_SAVE_PATH}/geneformer.npy', geneformer)
        np.save(f'{BASE_SAVE_PATH}/geneformer_gene.npy', geneformer_gene)

        return geneformer, geneformer_gene
    
    if data_type == 'prottrans':
        if os.path.exists(f'{BASE_SAVE_PATH}/prottrans.npy'):
            dat = np.load(f'{BASE_SAVE_PATH}/prottrans.npy')
            dat_gene = np.load(f'{BASE_SAVE_PATH}/prottrans_gene.npy')
            return dat, dat_gene
        
        dat = '/PRISME/dataset/protrans.pkl'
        with open(dat,'rb') as f:
            dat = pickle.load(f)
        dat_gene = list(dat.keys())
        dat = np.stack([np.array(dat[k]) for k in dat], axis=0)
        np.save(f'{BASE_SAVE_PATH}/{data_type}.npy', dat)
        np.save(f'{BASE_SAVE_PATH}/{data_type}_gene.npy', dat_gene)

        return dat, dat_gene
    
    if data_type == 'genept':
        if os.path.exists(f'{BASE_SAVE_PATH}/genept.npy'):
            genept = np.load(f'{BASE_SAVE_PATH}/genept.npy')
            genept_gene = np.load(f'{BASE_SAVE_PATH}/genept_gene.npy')
            return genept, genept_gene
        
        genept = '/PRISME/dataset/genept.pickle'
        genept = pickle.load(open(genept, 'rb'))
        genept_gene = list(genept.keys())
        genept = np.stack([np.array(genept[gene]) for gene in genept_gene], axis=0)

        np.save(f'{BASE_SAVE_PATH}/genept.npy', genept)
        np.save(f'{BASE_SAVE_PATH}/genept_gene.npy', genept_gene)

    if data_type == 'biolinkbert_summary':
        if os.path.exists(f'{BASE_SAVE_PATH}/biolinkbert_summary.npy'):
            biolinkbert_summary = np.load(f'{BASE_SAVE_PATH}/biolinkbert_summary.npy')
            biolinkbert_summary_gene = np.load(f'{BASE_SAVE_PATH}/biolinkbert_summary_gene.npy')
            return biolinkbert_summary, biolinkbert_summary_gene
        
        biolinkbert = '/PRISME/dataset/BioLinkBERT_summary.pkl'
        with open(biolinkbert,'rb') as f:
            biolinkbert = pickle.load(f)
        biolinkbert_gene = list(biolinkbert['symbol'])
        biolinkbert = np.stack([np.array(emb) for emb in biolinkbert['embedding']], axis=0)
        np.save(f'{BASE_SAVE_PATH}/biolinkbert_summary.npy', biolinkbert)
        np.save(f'{BASE_SAVE_PATH}/biolinkbert_summary_gene.npy', biolinkbert_gene)

        return biolinkbert, biolinkbert_gene

    if data_type == 'biolinkbert_genename':
        if os.path.exists(f'{BASE_SAVE_PATH}/biolinkbert_genename.npy'):
            biolinkbert_genename = np.load(f'{BASE_SAVE_PATH}/biolinkbert_genename.npy')
            biolinkbert_genename_gene = np.load(f'{BASE_SAVE_PATH}/biolinkbert_genename_gene.npy')
            return biolinkbert_genename, biolinkbert_genename_gene
        
        biolinkbert = '/PRISME/dataset/BioLinkBERT_name.pkl'
        with open(biolinkbert,'rb') as f:
            biolinkbert = pickle.load(f)
        biolinkbert_gene = list(biolinkbert['symbol'])
        biolinkbert = np.stack([np.array(emb) for emb in biolinkbert['embedding']], axis=0)
        np.save(f'{BASE_SAVE_PATH}/biolinkbert_genename.npy', biolinkbert)
        np.save(f'{BASE_SAVE_PATH}/biolinkbert_genename_gene.npy', biolinkbert_gene)

        return biolinkbert, biolinkbert_gene
    
    if data_type == 'struct2vec':
        if os.path.exists(f'{BASE_SAVE_PATH}/{data_type}.npy'):
            dat = np.load(f'{BASE_SAVE_PATH}/{data_type}.npy')
            dat_gene = np.load(f'{BASE_SAVE_PATH}/{data_type}_gene.npy')
            return dat, dat_gene
        
        dat = '/PRISME/dataset/struct2Vect/STRING_PPI_struc2vec_number_walks64_walk_length16_dim500.txt'
        id_map = '/PRISME/dataset/struct2Vect/node_list.txt'
        dat = pd.read_csv(dat, sep=' ', header=None, skiprows=1)
        dat_gene = dat.iloc[:, 0].values

        # read in id mapping
        id_map = pd.read_csv(id_map, sep='\t')
        dat_gene = [id_map[id_map['index'] == gene]['STRING_id'].values[0] for gene in dat_gene]
        dat_gene = [gene.split('.')[1] for gene in dat_gene] # ENSP id
        dat_gene_map = {i:gene for i,gene in enumerate(dat_gene)}


        ensembl2symbol = {}
        # batchly convert ensembl ids to gene symbols with batch size = 50
        for i in tqdm(range(0, len(dat_gene), 50)):
            ensembl2symbol.update(entrz_symbol.convert_ensp_to_gene_symbol(list(dat_gene[i:i+50])))

        dat_gene = [ensembl2symbol[gene] for gene in dat_gene]
        
        dat = np.array(dat.iloc[:, 1:])

        np.save(f'{BASE_SAVE_PATH}/{data_type}.npy', dat)
        np.save(f'{BASE_SAVE_PATH}/{data_type}_gene.npy', dat_gene)

        return dat, dat_gene
    
    if data_type == 'k2bTransE' or data_type == 'k2bMurE':
        if os.path.exists(f'{BASE_SAVE_PATH}/{data_type}.npy'):
            dat = np.load(f'{BASE_SAVE_PATH}/{data_type}.npy')
            dat_gene = np.load(f'{BASE_SAVE_PATH}/{data_type}_gene.npy')
            return dat, dat_gene
        if data_type=='k2bTransE':
            dat = '/PRISME/dataset/know2bio/transe_emb.pkl'
        elif data_type=='k2bMurE':
            dat = '/PRISME/dataset/know2bio/mure_emb.pkl'
        with open(dat,'rb') as f:
            dat = pickle.load(f)
        dat_gene = np.array([k for k in dat])
        dat = np.stack([np.array(dat[k]) for k in dat], axis=0)
        
        np.save(f'{BASE_SAVE_PATH}/{data_type}.npy', dat)
        np.save(f'{BASE_SAVE_PATH}/{data_type}_gene.npy', dat_gene)

        return dat, dat_gene    
        
def plot_cor_matrix(cor_matrix, dat_names, fig_name):
    # add transpose
    cor_matrix = cor_matrix + cor_matrix.T
    # set diagonal to 1
    np.fill_diagonal(cor_matrix, 1)
    
    plt.imshow(cor_matrix, cmap='viridis', interpolation='nearest')
    # put text in each cell
    for i in range(len(dat_names)):
        for j in range(len(dat_names)):
            plt.text(j, i, '{:.2f}'.format(cor_matrix[i, j]), ha='center', va='center', color='w')
    plt.xticks(np.arange(len(dat_names)), dat_names, rotation=90)
    plt.yticks(np.arange(len(dat_names)), dat_names)
    plt.tight_layout()

    plt.colorbar()
    plt.savefig(f'{RES_SAVE_PATH}/{fig_name}_cor_matrix.png', bbox_inches='tight')
    plt.show()
    plt.close()

def fit_svcca_model(dat1, dat2, permute=False):
    dat1 = dat1.T
    dat2 = dat2.T

    svcca_cor = cal_svcca(dat1, dat2)

    if permute:
        correlation_list = []

        for i in tqdm(range(n_sample)):
            dat1_perm = dat1[:, np.random.permutation(dat1.shape[1])]
            dat2_perm = dat2[:, np.random.permutation(dat2.shape[1])]
            correlation_list.append(cal_svcca(dat1_perm, dat2_perm))
        p_value = np.sum(np.array(correlation_list) > svcca_cor) / n_sample
        adj_svcca_cor = svcca_cor - np.mean(correlation_list)
    else:
        adj_svcca_cor = None
        p_value = None

    return svcca_cor, adj_svcca_cor, p_value


def cal_svcca(dat1, dat2):
    dat1 = dat1.astype(np.float64)
    dat2 = dat2.astype(np.float64)
    dat1 = dat1 - np.mean(dat1, axis=1, keepdims=True)
    dat2 = dat2 - np.mean(dat2, axis=1, keepdims=True)

    U1, s1, V1 = np.linalg.svd(dat1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(dat2, full_matrices=False)

    # keep top m singular values to have sum of singular values > 0.99
    m1 = np.sum(np.cumsum(np.absolute(s1)) / np.sum(np.absolute(s1)) < 0.99)
    m2 = np.sum(np.cumsum(np.absolute(s2)) / np.sum(np.absolute(s2)) < 0.99)

    svacts1 = np.dot(s1[:m1]*np.eye(m1), V1[:m1])
    svacts2 = np.dot(s2[:m2]*np.eye(m2), V2[:m2])

    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
    svcca_cor = np.mean(svcca_results['cca_coef1'])

    return svcca_cor


def remove_dup(dat, dat_gene):
    # remove duplication of genes in dat_gene, only keep the first one, and remove the corresponding row in dat
    dat_gene_unique = []
    dat_unique = []
    for i, gene in enumerate(dat_gene):
        if gene not in dat_gene_unique:
            dat_gene_unique.append(gene)
            dat_unique.append(dat[i, :])
    dat_unique = np.stack(dat_unique, axis=0)
    return dat_unique, dat_gene_unique


def main():
    svCCA_cor_matrix = np.zeros((len(dat_names), len(dat_names)))
    svCCA_cor_adj_matrix = np.zeros((len(dat_names), len(dat_names)))
    svCCA_p_value_matrix = np.zeros((len(dat_names), len(dat_names)))

    for dat_i in range(len(dat_names)-1):
        for dat_j in tqdm(range(dat_i+1, len(dat_names))):
            dat1, dat2, dat1_name, dat2_name = process_dat(dat_i, dat_j, dat_names)
            ## svCCA
            if fit_svcca:
                svcca_cor, adj_svcca_cor, p_value = fit_svcca_model(dat1, dat2, permute=PERMUTE)
                svCCA_cor_matrix[dat_i, dat_j] = svcca_cor
                svCCA_cor_adj_matrix[dat_i, dat_j] = adj_svcca_cor
                svCCA_p_value_matrix[dat_i, dat_j] = p_value
    
    if fit_svcca:
        plot_cor_matrix(svCCA_cor_matrix, dat_names, 'svCCA')
        np.save(f'{RES_SAVE_PATH}/svCCA_cor_matrix.npy', svCCA_cor_matrix)
        if PERMUTE:
            plot_cor_matrix(svCCA_cor_adj_matrix, dat_names, 'svCCA_adj')
            np.save(f'{RES_SAVE_PATH}/svCCA_cor_adj_matrix.npy', svCCA_cor_adj_matrix)
            plot_cor_matrix(svCCA_p_value_matrix, dat_names, 'svCCA_p_value')
            np.save(f'{RES_SAVE_PATH}/svCCA_p_value_matrix.npy', svCCA_p_value_matrix)


if __name__ == '__main__':
    main()
