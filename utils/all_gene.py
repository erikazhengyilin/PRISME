from compare_all import read_dat, dat_names

dat_genes = []
for dat_i in range(len(dat_names)):
    dat, dat_gene = read_dat(dat_names[dat_i])
    dat_genes.append(dat_gene)

# get intersection
ALL_INTER_GENE = set(dat_genes[0])
for dat_gene in dat_genes:
    ALL_INTER_GENE = ALL_INTER_GENE.intersection(set(dat_gene))

print(len(ALL_INTER_GENE))