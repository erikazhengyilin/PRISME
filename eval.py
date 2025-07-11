import pandas as pd
import numpy as np
import os
from compare_all import dat_names
from utils.all_gene import ALL_INTER_GENE 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm
import xgboost as xgb
from scipy.stats import loguniform
from pycaret.classification import *
from pycaret.classification import ClassificationExperiment

RES_SAVE_PATH = "result"
BASE_SAVE_PATH = ""
dat_names = ['gene2vec', 'omics', 'geneformer', 'prottrans', 'genept', 'biolinkbert_summary', 'struct2vec', 'k2bTransE', 'k2bMurE', 'UGE']

#task_names =  ['loc','ggipnn','dosage','ppi','GO', 'dis_gene']
task_names =  ['dis_gene']
ML_MODEL = 'logistic'

EVAL_ML = True
EVAL_CCA = True

def main():
    for task_name in task_names:
        for dat_name in dat_names:
            dat = np.load(os.path.join(BASE_SAVE_PATH, f'dataset/{dat_name}.npy'), allow_pickle=True)
            dat_gene = np.load(os.path.join(BASE_SAVE_PATH, f'dataset/{dat_name}_gene.npy'), allow_pickle=True)
            task_eval_dict[task_name](dat, dat_gene, dat_name)
            print(dat_name, task_name)        


def eval_ppi(dat, dat_gene, dat_name):
    # load data
    ppi_dat = pd.read_csv(os.path.join(BASE_SAVE_PATH, '/home/erikayilin/uge_capybara/benchmark_datasets/ppi.tsv'), sep='\t')
    ppi_dat = ppi_dat.rename(columns={'gene': 'gene_symbol'})

    # get unique genes from PPI
    ppi_dat_gene = list(set(ppi_dat['protein1']).union(set(ppi_dat['protein2'])))
    ppi_dat_gene = [gene for gene in ppi_dat_gene if gene in ALL_INTER_GENE]

    gene_to_index = {gene: idx for idx, gene in enumerate(dat_gene)}
    dim = dat.shape[1]

    ys = []
    Xs = []

    # Counters for diagnostics
    total_pairs = 0
    g1_random, g2_random = 0, 0

    for row in tqdm(ppi_dat.itertuples(index=False), total=len(ppi_dat)):
        g1, g2 = row.protein1, row.protein2
        if g1 not in ALL_INTER_GENE or g2 not in ALL_INTER_GENE:
            continue

        total_pairs += 1

        if g1 in gene_to_index:
            v1 = dat[gene_to_index[g1]]
        else:
            v1 = np.random.normal(0, 1, dim)
            g1_random += 1

        if g2 in gene_to_index:
            v2 = dat[gene_to_index[g2]]
        else:
            v2 = np.random.normal(0, 1, dim)
            g2_random += 1

        Xs.append(v1 * v2)
        ys.append(row.combined_score)

    Xs = np.array(Xs)
    ys = np.array(ys)

    # Print diagnostic info
    print(f"Total valid PPI pairs evaluated: {total_pairs}")
    print(f"Protein1: {g1_random} random fills, {total_pairs - g1_random} from dat_gene")
    print(f"Protein2: {g2_random} random fills, {total_pairs - g2_random} from dat_gene")

    save_name = 'PPI_' + dat_name
    pycaret_eval(Xs, ys, save_name)


def eval_go(dat, dat_gene, dat_name):
    # load data
    go_dat = pd.read_csv(os.path.join(BASE_SAVE_PATH, '/home/erikayilin/uge_capybara/benchmark_datasets/go_BP.tsv'), sep='\t')
    go_dat = go_dat.rename(columns={'gene': 'gene_symbol'})

    go_dat = go_dat.dropna(subset=['gene_symbol'])
    go_dat = go_dat[go_dat['gene_symbol'].isin(ALL_INTER_GENE)]

    gene_to_index = {gene: idx for idx, gene in enumerate(dat_gene)}
    dim = dat.shape[1]
    go_scores = {}

    for go in tqdm(go_dat.columns[1:]):
        go_dat_sub = go_dat[['gene_symbol', go]].copy()
        go_dat_sub = go_dat_sub.sort_values('gene_symbol').reset_index(drop=True)

        Xs = []
        ys = []

        found_count = 0
        random_count = 0
        random_genes = set()

        for row in go_dat_sub.itertuples(index=False):
            gene = row.gene_symbol
            label = getattr(row, go)

            if gene in gene_to_index:
                vec = dat[gene_to_index[gene]]
                found_count += 1
            else:
                vec = np.random.normal(0, 1, dim)
                random_count += 1
                random_genes.add(gene)

            Xs.append(vec)
            ys.append(label)

        Xs = np.array(Xs)
        ys = np.array(ys)

        print(f"[{go}] Total genes: {len(Xs)} | Found: {found_count} | Random: {random_count} | Unique random genes: {len(random_genes)}")

        save_name = 'go_' + go + '_' + dat_name
        pycaret_eval(Xs, ys, save_name)

    return go_scores


def eval_loc(dat, dat_gene, dat_name):
    # load data
    loc = pd.read_csv(os.path.join(BASE_SAVE_PATH, '/home/erikayilin/uge_capybara/benchmark_datasets/deeploc.tsv'), sep='\t')

    loc_list = loc.columns[:-1]
    loc = loc.dropna(subset=['gene'])
    loc = loc[loc['gene'].isin(ALL_INTER_GENE)]

    gene_to_index = {gene: idx for idx, gene in enumerate(dat_gene)}
    dim = dat.shape[1]

    for lo in tqdm(loc_list):
        ys = []
        Xs = []

        found_count = 0
        random_count = 0
        random_genes = set()

        for row in loc.itertuples(index=False):
            gene = row.gene
            label = getattr(row, lo)

            if not (label == 0 or label == 1):
                continue

            if gene in gene_to_index:
                vec = dat[gene_to_index[gene]]
                found_count += 1
            else:
                vec = np.random.normal(0, 1, dim)
                random_count += 1
                random_genes.add(gene)

            Xs.append(vec)
            ys.append(label)

        ys = np.array(ys)
        Xs = np.array(Xs)

        if sum(ys == 1) == 0 or sum(ys == 0) == 0:
            continue  # skip unbalanced label classes

        if lo == 'Lysosome/Vacuole':
            lo = 'Lysosome|Vacuole'

        print(f"[{lo}] Total samples: {len(ys)} | Found: {found_count} | Random: {random_count} | Unique random genes: {len(random_genes)}")

        save_name = 'loc_' + lo + '_' + dat_name
        pycaret_eval(Xs, ys, save_name)


def eval_dis_gene(dat, dat_gene, dat_name):
    # load data
    dis_gene = pd.read_csv(os.path.join(BASE_SAVE_PATH, '/home/erikayilin/uge_capybara/benchmark_datasets/dis_gene_final.txt'), sep='\t')
    dis_gene = dis_gene.drop_duplicates(subset=['gene_symbol', 'disease'])
    dis_gene = dis_gene.dropna(subset=['gene_symbol', 'disease', 'target'])
    dis_gene = dis_gene[dis_gene['gene_symbol'].isin(ALL_INTER_GENE)]

    gene_to_index = {gene: idx for idx, gene in enumerate(dat_gene)}
    dim = dat.shape[1]
    dis_scores = {}

    print('evaluate disease separately')
    for dis in tqdm(dis_gene['disease'].unique()):
        dis_gene_sub = dis_gene[dis_gene['disease'] == dis]
        dis_gene_sub = dis_gene_sub.sort_values('gene_symbol').reset_index(drop=True)

        Xs = []
        ys = []

        found_count = 0
        random_count = 0
        random_genes = set()

        for row in dis_gene_sub.itertuples(index=False):
            gene = row.gene_symbol
            label = row.target

            if gene in gene_to_index:
                vec = dat[gene_to_index[gene]]
                found_count += 1
            else:
                vec = np.random.normal(0, 1, dim)
                random_count += 1
                random_genes.add(gene)

            Xs.append(vec)
            ys.append(label)

        Xs = np.array(Xs)
        ys = np.array(ys)

        if sum(ys == 1) == 0 or sum(ys == 0) == 0:
            continue

        print(f"[{dis}] Total genes: {len(ys)} | Found: {found_count} | Random: {random_count} | Unique random genes: {len(random_genes)}")

        save_name = 'dis_gene_' + dis + '_' + dat_name
        pycaret_eval(Xs, ys, save_name)

    return dis_scores


def eval_dosage(dat, dat_gene, dat_name):
    # load data
    sensitive = pd.read_csv(os.path.join(BASE_SAVE_PATH, '/home/erikayilin/uge_capybara/benchmark_datasets/dosage_sensitive.txt'), header=None)
    insensitive = pd.read_csv(os.path.join(BASE_SAVE_PATH, '/home/erikayilin/uge_capybara/benchmark_datasets/dosage_insensitive.txt'), header=None)

    sensitive = list(sensitive[0])
    insensitive = list(insensitive[0])
    sensitive = [gene for gene in sensitive if gene in ALL_INTER_GENE]
    insensitive = [gene for gene in insensitive if gene in ALL_INTER_GENE]

    gene_to_index = {gene: idx for idx, gene in enumerate(dat_gene)}
    dim = dat.shape[1]

    sen_dat = []
    sen_found = 0
    sen_random = 0
    sen_random_genes = set()

    for gene in sensitive:
        if gene in gene_to_index:
            vec = dat[gene_to_index[gene]]
            sen_found += 1
        else:
            vec = np.random.normal(0, 1, dim)
            sen_random += 1
            sen_random_genes.add(gene)
        sen_dat.append(vec)

    insen_dat = []
    insen_found = 0
    insen_random = 0
    insen_random_genes = set()

    for gene in insensitive:
        if gene in gene_to_index:
            vec = dat[gene_to_index[gene]]
            insen_found += 1
        else:
            vec = np.random.normal(0, 1, dim)
            insen_random += 1
            insen_random_genes.add(gene)
        insen_dat.append(vec)

    Xs = np.concatenate([sen_dat, insen_dat], axis=0)
    ys = np.concatenate([np.ones(len(sen_dat)), np.zeros(len(insen_dat))], axis=0)

    print(f"[Dosage Sensitive] Total: {len(sen_dat)} | Found: {sen_found} | Random: {sen_random} | Unique Random Genes: {len(sen_random_genes)}")
    print(f"[Dosage Insensitive] Total: {len(insen_dat)} | Found: {insen_found} | Random: {insen_random} | Unique Random Genes: {len(insen_random_genes)}")

    save_name = 'dosage_' + dat_name
    pycaret_eval(Xs, ys, save_name)


def eval_ggipnn(dat, dat_gene, dat_name):
    # Load gene pairs and labels
    ggi = pd.read_csv(os.path.join(BASE_SAVE_PATH, '/home/erikayilin/uge_capybara/benchmark_datasets/GGIPNN/ggipnn_gene_pairs.txt'), sep='\t', header=None)
    label = pd.read_csv(os.path.join(BASE_SAVE_PATH, '/home/erikayilin/uge_capybara/benchmark_datasets/GGIPNN/ggipnn_labels.txt'), sep='\t', header=None)

    # Add labels
    ggi['y'] = label[0].values

    # Keep only gene pairs where both genes are in ALL_INTER_GENE
    ggi = ggi[ggi[0].isin(ALL_INTER_GENE) & ggi[1].isin(ALL_INTER_GENE)].reset_index(drop=True)

    gene_to_index = {gene: idx for idx, gene in enumerate(dat_gene)}
    dim = dat.shape[1]

    ys = ggi['y'].values
    Xs = np.empty((len(ggi), dim))

    # Counters
    g1_random = 0
    g2_random = 0
    g1_random_genes = set()
    g2_random_genes = set()

    for i, row in tqdm(enumerate(ggi.itertuples(index=False)), total=len(ggi)):
        g1 = row[0]
        g2 = row[1]

        if g1 in gene_to_index:
            v1 = dat[gene_to_index[g1]]
        else:
            v1 = np.random.normal(0, 1, dim)
            g1_random += 1
            g1_random_genes.add(g1)

        if g2 in gene_to_index:
            v2 = dat[gene_to_index[g2]]
        else:
            v2 = np.random.normal(0, 1, dim)
            g2_random += 1
            g2_random_genes.add(g2)

        Xs[i] = v1 * v2

    print(f"[GGIPNN] Total pairs: {len(ggi)}")
    print(f"Protein1: Random fills = {g1_random}, Unique random genes = {len(g1_random_genes)}")
    print(f"Protein2: Random fills = {g2_random}, Unique random genes = {len(g2_random_genes)}")

    save_name = 'ggipnn_' + dat_name
    pycaret_eval(Xs, ys, save_name)


def pycaret_eval(Xs, ys, save_name):
    n_dim = Xs.shape[1]
    dat = np.concatenate([Xs, ys.reshape(-1,1)], axis=1)
    columns = ['dim'+str(i) for i in range(n_dim)] + ['target']
    dat = pd.DataFrame(dat, columns=columns)
    s = setup(dat, target="target",  session_id = 123, use_gpu = True)
    add_metric('auprc', 'auprc', average_precision_score)

    exp = ClassificationExperiment()
    exp.setup(dat, target = 'target', session_id = 123)
    best = compare_models(sort='auprc', n_select=1, include = ['lr'])#, 'knn', 'nb', 'rf', 'mlp'])
    df = pull()
    df.to_csv(os.path.join(RES_SAVE_PATH, save_name+'_pycaret_results.tsv'), sep='\t')
    save_model(best, save_name)
    

def preprocess(eval_dat, dat, dat_gene): 
    # remove duplicate genes
    eval_dat = eval_dat[eval_dat['gene_symbol'].isin(ALL_INTER_GENE)]
    dat = dat[np.array([i for i,gene in enumerate(dat_gene) if gene in ALL_INTER_GENE]), :]
    dat_gene = [gene for gene in dat_gene if gene in ALL_INTER_GENE]

    # keep interected genes
    eval_dat_gene_list = eval_dat['gene_symbol'].unique()
    inter_gene = np.intersect1d(eval_dat_gene_list, dat_gene)
    print('number of intersected genes ', len(inter_gene))
    
    eval_dat = eval_dat[eval_dat['gene_symbol'].isin(inter_gene)]
    dat = dat[np.array([i for i,gene in enumerate(dat_gene) if gene in inter_gene]), :]
    dat_gene = [gene for gene in dat_gene if gene in inter_gene]

    # sort genes
    dat = dat[np.argsort(dat_gene), :]
    return eval_dat, dat, dat_gene
    

def cross_validation(Xs, ys, n_fold=5):
    # assign fold
    idx = np.arange(len(ys))
    np.random.shuffle(idx)
    fold = idx%n_fold

    auprcs = []
    shuffle_auprcs = []
    # five fold cross validation
    for i in range(n_fold):
        idx_train = fold!=i
        idx_test = fold==i
        X_train = Xs[idx_train, :]
        X_test = Xs[idx_test, :]
        y_train = ys[idx_train]
        y_test = ys[idx_test]
        
        # convert to binary
        y_train = np.array(y_train==True, dtype=int)
        y_test = np.array(y_test==True, dtype=int)
        
        accuracy, auprc = do_ml(X_train, X_test, y_train, y_test, ml_model=ML_MODEL)

        np.random.shuffle(y_train)
        accuracy, shuffle_auprc = do_ml(X_train, X_test, y_train, y_test, ml_model=ML_MODEL)
        auprcs.append(auprc)
        shuffle_auprcs.append(shuffle_auprc)
    return np.mean(auprcs), np.mean(shuffle_auprcs)


def do_ml(X_train, X_test, y_train, y_test, ml_model='xgboost'):
    if ml_model=='xgboost':
        param_dist = {
            "reg_lambda": loguniform(1e-2, 1e5), 
            "reg_alpha": loguniform(1e-2, 1e5)
        }
        reg = xgb.XGBClassifier(tree_method="hist")
        random_search = RandomizedSearchCV(
            reg, param_distributions=param_dist, n_iter=50, refit = True
        )
        
        mod = random_search.fit(
                X_train, y_train
            )

        y_pred = mod.predict_proba(
                X_test
            )
        
        accuracy, auprc = evaluate(y_pred, y_test)
    
    if ml_model=='logistic':
        clf = LogisticRegression(random_state=0).fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)[:,1]
        accuracy, auprc = evaluate(y_pred, y_test)
    
    
    return accuracy, auprc


def evaluate(y_pred, y_test):
    # print('accuracy', np.mean(y_pred==y_test))
    accuracy = np.mean(y_pred==y_test)

    # confusion matrix
    # print('confusion matrix' )
    # print(confusion_matrix(y_test, y_pred>0.5))

    # AUPRC
    # print('AUPRC')
    auprc = average_precision_score(y_test, y_pred)
    # print(auprc)

    return accuracy, auprc


task_eval_dict = {'dis_gene': eval_dis_gene, 'GO': eval_go, 
                  'dosage':eval_dosage,'ggipnn':eval_ggipnn,'loc':eval_loc,
                  'ppi': eval_ppi}


if __name__ == "__main__":
    main()
