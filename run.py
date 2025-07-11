# run autoencoder on concantenated data
import torch
from utils.model import DeepAutoencoder
import os
import numpy as np
from tqdm import tqdm
from compare_all import read_dat, dat_names
import datetime

# edit path here
BASE_SAVE_PATH = "dataset"
MODEL_SAVE_PATH = "dataset"

current_time = datetime.datetime.now()
time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")

# tb_leave_out = ['biolinkbert_summary']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    suffix = f'UGE~ALL~512_{time_string}'
    dats, dat_genes = [], []
    # read data
    for dat_name in dat_names:
        if dat_name in ['autoencoder']:
            continue
        dat, dat_gene = read_dat(dat_name)
        dats.append(dat)
        dat_genes.append(dat_gene)

    # get intersection of genes
    genes = set(dat_genes[0])
    for dat_gene in dat_genes[1:]:
        genes = genes.intersection(set(dat_gene))
    genes = list(genes)
    print('number of genes: {}'.format(len(genes)))

    # concatenate data by gene
    dat_concat = []
    dim_list = []
    for i, dat in enumerate(dats):
        dat = dat[np.array([i for i,gene in enumerate(dat_genes[i]) if gene in genes]), :]
        dat_gene = [gene for gene in dat_genes[i] if gene in genes]
        # sort dat by gene
        dat = dat[np.argsort(dat_gene), :]
        # normalize
        dat = (dat - np.mean(dat, axis=0)) / np.std(dat, axis=0)
        dat_concat.append(dat)
        dim_list.append(dat.shape[1])

    # sort genes
    genes = np.array(genes)
    genes = genes[np.argsort(genes)]

    dat_concat = np.concatenate(dat_concat, axis=1)
    print(dat_concat.shape)

    # train autoencoder
    model = DeepAutoencoder(dat_concat.shape[1], 2048, 1024, 512).to(device)
    criterion = torch.nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 100

    # build a dataloader
    dataloader = torch.utils.data.DataLoader(dat_concat, batch_size=32, shuffle=True)
    emb_weight = get_weight(dim_list=dim_list, batch_size=32)
    emb_weight = torch.from_numpy(emb_weight).to(device).float()
    best_loss = 100
    patient_count = 0
    for epoch in range(n_epochs):
        losses = 0
        for dat in tqdm(dataloader):
            dat = dat.float().to(device)
            output = model(dat)
            emb_weight_ = torch.tile(emb_weight, (dat.shape[0],1))
            loss = criterion(output, torch.Tensor(dat)) * emb_weight_
            loss = torch.mean(loss)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
        if losses/dat_concat.shape[0] < best_loss:
            patient_count = 0
            best_loss = losses/dat_concat.shape[0]
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f'{suffix}.pth'))
        else:
            patient_count += 1
            if patient_count > 3:
                break
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, n_epochs, losses/dat_concat.shape[0]))

    # save model
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f'{suffix}.pth'))

    # get latent space
    model.eval()
    dataloader = torch.utils.data.DataLoader(dat_concat, batch_size=32, shuffle=False)
    latents = []
    with torch.no_grad():
        for dat in tqdm(dataloader):
            dat = dat.float().to(device)
            latent = model.encoder(dat)
            latents.append(latent)
    latents = torch.cat(latents, dim=0).cpu().numpy()
    # save latent space
    np.save(os.path.join(BASE_SAVE_PATH, f'{suffix}.npy'), latents)
    # save genes
    np.save(os.path.join(BASE_SAVE_PATH, f'{suffix}_gene.npy'), genes)


def get_weight(dim_list, batch_size=32):
    total = sum(dim_list)
    weight = [total/dim for dim in dim_list]
    weight = np.array(weight)
    weight = weight / np.sum(weight)
    weights = np.concatenate([np.repeat(w, dim_list[i]) for i,w in enumerate(weight)])
    weights = np.expand_dims(weights, axis=0)
    return weights

if __name__ == '__main__':
    main()