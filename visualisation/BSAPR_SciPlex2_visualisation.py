import gzip
import requests
from anndata import AnnData
import os
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from BSAPR_model import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
adata = sc.read('/Users/hanwenxing/PycharmProjects/perturbation/SciPlex2_new.h5ad')

torch.manual_seed(3141592)
# load data:
my_conditioner = pd.read_csv("SciPlex2_perturbation.csv", index_col=0)
my_conditioner = my_conditioner.drop('Vehicle', axis=1)  # TODO: or retaining it
cond_name = list(my_conditioner.columns)
my_conditioner = torch.tensor(my_conditioner.to_numpy() * 1.0, dtype=torch.float)
my_conditioner = torch.sqrt(my_conditioner)

my_observation = pd.read_csv("'SciPlex2.csv", index_col=0)
print(my_observation.shape)
my_observation = torch.tensor(my_observation.to_numpy() * 1.0, dtype=torch.float)

download_path = "./srivatsan_2019_sciplex2"
gene_metadata = pd.read_csv(
    os.path.join(
        download_path,
        "GSM4150377_sciPlex2_A549_Transcription_Modulators_gene.annotations.txt.gz",
    ),
    sep="\t",
    header=None,
    index_col=0,
)
gene_name = [gene_metadata.loc[i].values[0] for i in list(adata.var.index)]

#
# gene_name_lookup = pd.read_csv('LUHMES_data/LUHMES_unfiltered_gene_lookup.csv', index_col=0)
# gene_name_lookup = {gene_name_lookup.iloc[i].unfiltered_genes: gene_name_lookup.iloc[i].feature_name_lookup_unfiltered for i in range(gene_name_lookup.shape[0])}
# gene_name = [gene_name_lookup[i] for i in gene_name]
#

my_cell_info = pd.read_csv("SciPlex2_cell_info.csv", index_col=0)
my_cell_info.n_genes = my_cell_info.n_genes/my_cell_info.n_counts
my_cell_info.n_counts = np.log(my_cell_info.n_counts)
cell_info_names = list(my_cell_info.columns)
my_cell_info = torch.tensor(my_cell_info.to_numpy() * 1.0, dtype=torch.float)

# design the training process:
start = time.time()
output_dim = my_observation.shape[1]
sample_size = my_observation.shape[0]
hidden_node = 1000  # or 1000
hidden_layer_1 = 4
hidden_layer_2 = 4
conditioner_dim = my_conditioner.shape[1]
cell_info_dim = my_cell_info.shape[1]

lr_parametric = 1e-3  # 7e-4 works, 2e-3 kinda works  # 1500, 1e-3, 0.005, 40, 0.37
nu_1, nu_2, nu_3, nu_4, nu_5, nu_6 = torch.tensor([1., 1e-4, 1., 1e-4, 1., 1e-4]).to(device)
tau = torch.tensor(1.).to(device)
KL_factor = torch.tensor(1.).to(device)

# generate synthetic data
torch.manual_seed(314159)

my_dataset = MyDataSet(observation=my_observation, conditioner=my_conditioner, cell_info=my_cell_info)

testing_idx = set(np.random.choice(a=range(my_observation.shape[0]), size=my_observation.shape[0]//8, replace=False))
training_idx = list(set(range(my_observation.shape[0])) - testing_idx)
testing_idx = list(testing_idx)
training_idx_sampler = torch.utils.data.SubsetRandomSampler(training_idx)
training_loader = torch.utils.data.DataLoader(my_dataset, batch_size=100, sampler=training_idx_sampler)
test_observation = my_observation[testing_idx].to(device)
test_conditioner = my_conditioner[testing_idx].to(device)
test_cell_info = my_cell_info[testing_idx].to(device)

parametric_model = BSAPR_Gaussian(conditioner_dim=conditioner_dim, output_dim=output_dim, base_dim=cell_info_dim,
                               data_size=sample_size, hidden_node=hidden_node, hidden_layer_1=hidden_layer_1,
                               hidden_layer_2=hidden_layer_2, tau=tau)
parametric_model = parametric_model.to(device)


parametric_model.load_state_dict(torch.load('BSAPR_SciPlex2_Gaussian.pt'))
p = 0.95
top=50
# how does each condition affect different genes?
unique_conditions = torch.unique(my_conditioner, dim=0)
perturb_level, _, _, _, logit_p, _, _ = parametric_model(unique_conditions, None)
estimated_inclusion_prob = F.sigmoid(logit_p).detach().cpu().numpy()
estimated_inclusion = estimated_inclusion_prob > p
my_gene_name = np.array(gene_name)[estimated_inclusion.sum(axis=0) > 0]
estimated_inclusion_prob = estimated_inclusion_prob[:, estimated_inclusion.mean(axis=0) > 0]
estimated_pert = perturb_level.detach().cpu().numpy() * estimated_inclusion
estimated_pert = estimated_pert[:, estimated_inclusion.mean(axis=0) > 0]
top_id = np.argsort((estimated_pert**2).sum(axis=0))[::-1][:top]
estimated_pert = estimated_pert[:, top_id]
my_gene_name = my_gene_name[top_id]

unique_conditions = unique_conditions.numpy()
my_yticks = ['' for _ in range(unique_conditions.shape[0])]
for i in range(unique_conditions.shape[0]):
    if np.all(unique_conditions[i] == 0):
        my_yticks[i] = 'Vehicle_1.0'
    else:
        my_yticks[i] = np.array(cond_name)[unique_conditions[i] != 0.][0] + '_' + str(np.round((unique_conditions[i][unique_conditions[i] != 0][0])**2,3))

fig, ax1 = plt.subplots(1,1)

import matplotlib.colors as colors
negatives = estimated_pert.min()
positives = estimated_pert.max()

num_neg_colors = int(256 / (positives - negatives) * (-negatives))
num_pos_colors = 256 - num_neg_colors
cmap_BuRd = plt.cm.RdBu_r
colors_2neg_4pos = [cmap_BuRd(0.5*c/num_neg_colors) for c in range(num_neg_colors)] +\
                   [cmap_BuRd(1-0.5*c/num_pos_colors) for c in range(num_pos_colors)][::-1]
cmap_2neg_4pos = colors.LinearSegmentedColormap.from_list('cmap_2neg_4pos', colors_2neg_4pos, N=256)

# im = ax1.imshow(estimated_pert[1:], norm=norm, cmap='RdBu_r')
im = ax1.imshow(estimated_pert[1:], cmap=cmap_2neg_4pos)
for i in [6.5, 13.5, 20.5]:
    ax1.axline((i, i), slope=0, alpha=0.4, c='k')
ax1.set_xticks(np.arange(len(my_gene_name)), my_gene_name, rotation=90)
ax1.set_yticks(np.arange(len(my_yticks)-1), my_yticks[1:])
ax1.set_title('Estimated perturbation effects')
ticks = np.append(np.arange(-2., 0., .5), np.arange(0, 3.001, 0.75))
fig.colorbar(im, ax=ax1, ticks=ticks)
fig.set_size_inches(12, 8)
fig.tight_layout()
plt.savefig('BSAPR_heatmap_SciPlex2.png')
plt.close()


# now check fitted results
predicted_mu_mean, predicted_mu_var, predicted_gamma_mean, predicted_gamma_var, \
    logit_p, logit_p_log_var, predicted_base_mean = parametric_model(my_conditioner, my_cell_info)
estimated_base_mean = predicted_base_mean.detach().cpu().numpy()
estimated_perturbed_mean = (F.sigmoid(logit_p) * predicted_mu_mean).detach().cpu().numpy()
estimated_total_mean = (estimated_perturbed_mean + estimated_base_mean)
estimated_perturbed_var = logexpp1(parametric_model.base_log_var + (F.sigmoid(logit_p) * predicted_gamma_mean)).detach().cpu().numpy()


avg_unique_pert = np.zeros((len(adata.obs.drug_dose_name.unique()), output_dim))
avg_obs = np.zeros((len(adata.obs.drug_dose_name.unique()), output_dim))
unique_pert = np.array(adata.obs.drug_dose_name.unique())
baseline_id = 0
for i, name in enumerate(list(adata.obs.drug_dose_name.unique())):
    if name == 'Vehicle_1.0':
        baseline_id = i
        print(i)
    avg_unique_pert[i] = estimated_total_mean[np.array(adata.obs.drug_dose_name == name)].mean(axis=0)
    avg_obs[i] = my_observation[np.array(adata.obs.drug_dose_name == name)].mean(axis=0).numpy()
avg_obs = np.delete(avg_obs, baseline_id, 0)
avg_unique_pert = np.delete(avg_unique_pert, baseline_id, 0)

fig, axs = plt.subplots(1, 2)
# axs[0].scatter(my_observation.cpu().numpy().ravel(), estimated_total_mean.ravel(), alpha=0.15)
# axs[0].axline((1, 1), slope=1, c='r', alpha=0.5, linestyle='--')
# axs[0].set_xlabel('Predicted perturbed responses')
# axs[0].set_ylabel('Observed responses')
# axs[0].set_title('SciPlex2, Predicted vs Observed')

axs[0].scatter(avg_obs.ravel(), avg_unique_pert.ravel(), alpha=0.15)
axs[0].axline((1, 1), slope=1, c='r', alpha=0.5, linestyle='--')
axs[0].set_xlabel('Averaged prediction for each unique perturbation')
axs[0].set_ylabel('Averaged observation for each unique perturbation')
axs[0].set_title('Averaged prediction vs Averaged observation, BSAPR')


CPA_avg_pred = pd.read_csv('CPA_SciPlex2_avg_pred.csv', index_col=0).to_numpy().ravel()
axs[1].scatter(avg_obs.ravel(), CPA_avg_pred, alpha=0.15)
axs[1].axline((1, 1), slope=1, c='r', alpha=0.5, linestyle='--')
axs[1].set_xlabel('Averaged prediction for each unique perturbation')
axs[1].set_ylabel('Averaged observation for each unique perturbation')
axs[1].set_title('Averaged prediction vs Averaged observation, CPA')
fig.set_size_inches(12, 6)
fig.tight_layout()
plt.savefig('BSAPR_vs_CPA_SciPlex2.png')
plt.close()

print('CPA R2 = {}'.format(np.round(np.corrcoef(avg_obs.ravel(), CPA_avg_pred)[0, 1], 5)))
print('BSAPR R2 = {}'.format(np.round(np.corrcoef(avg_obs.ravel(), avg_unique_pert.ravel())[0,1], 5)))


# visualizing the ZIP model

download_path = "./srivatsan_2019_sciplex2"
torch.manual_seed(3141592)
# load data:
my_conditioner = pd.read_csv("SciPlex2_perturbation.csv", index_col=0)
my_conditioner = my_conditioner.drop('Vehicle', axis=1)  # TODO: or retaining it
cond_name = list(my_conditioner.columns)
my_conditioner = torch.tensor(my_conditioner.to_numpy() * 1.0, dtype=torch.float)
my_conditioner = torch.sqrt(my_conditioner)

adata = sc.read('/Users/hanwenxing/PycharmProjects/perturbation/SciPlex2_new.h5ad')
my_observation = adata.layers['counts']
print(my_observation.shape)
my_observation = torch.tensor(my_observation * 1.0, dtype=torch.float)


gene_metadata = pd.read_csv(
        "GSM4150377_sciPlex2_A549_Transcription_Modulators_gene.annotations.txt.gz", sep="\t", header=None, index_col=0)
gene_name = [gene_metadata.loc[i].values[0] for i in list(adata.var.index)]


my_cell_info = pd.read_csv("SciPlex2_cell_info.csv", index_col=0)
my_cell_info.n_genes = my_cell_info.n_genes/my_cell_info.n_counts
my_cell_info.n_counts = np.log(my_cell_info.n_counts)
cell_info_names = list(my_cell_info.columns)
my_cell_info = torch.tensor(my_cell_info.to_numpy() * 1.0, dtype=torch.float)


# design the training process:
start = time.time()
output_dim = my_observation.shape[1]
sample_size = my_observation.shape[0]
hidden_node = 1000  # or 1000
hidden_layer_1 = 4
hidden_layer_2 = 4
conditioner_dim = my_conditioner.shape[1]
cell_info_dim = my_cell_info.shape[1]

lr_parametric = 1e-3  # 7e-4 works, 2e-3 kinda works  # 1500, 1e-3, 0.005, 40, 0.37
nu_1, nu_2, nu_3, nu_4 = torch.tensor([5., 1e-4, 1., 1e-4]).to(device)
tau = torch.tensor(1.).to(device)

# generate synthetic data
torch.manual_seed(314159)

my_dataset = MyDataSet(observation=my_observation, conditioner=my_conditioner, cell_info=my_cell_info)

testing_idx = set(np.random.choice(a=range(my_observation.shape[0]), size=my_observation.shape[0]//8, replace=False))
training_idx = list(set(range(my_observation.shape[0])) - testing_idx)
testing_idx = list(testing_idx)
training_idx_sampler = torch.utils.data.SubsetRandomSampler(training_idx)
training_loader = torch.utils.data.DataLoader(my_dataset, batch_size=100, sampler=training_idx_sampler)
test_observation = my_observation[testing_idx].to(device)
test_conditioner = my_conditioner[testing_idx].to(device)
test_cell_info = my_cell_info[testing_idx].to(device)

parametric_model = BSAPR_ZIP(conditioner_dim=conditioner_dim, output_dim=output_dim, base_dim=cell_info_dim,
                               data_size=sample_size, hidden_node=hidden_node, hidden_layer_1=hidden_layer_1,
                               hidden_layer_2=hidden_layer_2, tau=tau)
parametric_model = parametric_model.to(device)
parametric_model.load_state_dict(torch.load('BSAPR_SciPlex2_ZIP.pt'))
p = 0.95
top=50
# how does each condition affect different genes?
unique_conditions = torch.unique(my_conditioner, dim=0)
perturb_level, _, base_mean, logit_p, _ = parametric_model(unique_conditions, None)
estimated_inclusion_prob = F.sigmoid(logit_p).detach().cpu().numpy()
estimated_inclusion = estimated_inclusion_prob > p
my_gene_name = np.array(gene_name)[estimated_inclusion.sum(axis=0) > 0]
estimated_inclusion_prob = estimated_inclusion_prob[:, estimated_inclusion.mean(axis=0) > 0]
estimated_pert = perturb_level.detach().cpu().numpy() * estimated_inclusion
estimated_pert = estimated_pert[:, estimated_inclusion.mean(axis=0) > 0]
top_id = np.argsort((estimated_pert**2).sum(axis=0))[::-1][:top]
estimated_pert = estimated_pert[:, top_id]
my_gene_name = my_gene_name[top_id]

unique_conditions = unique_conditions.numpy()
my_yticks = ['' for _ in range(unique_conditions.shape[0])]
for i in range(unique_conditions.shape[0]):
    if np.all(unique_conditions[i] == 0):
        my_yticks[i] = 'Vehicle_1.0'
    else:
        my_yticks[i] = np.array(cond_name)[unique_conditions[i] != 0.][0] + '_' + str(np.round((unique_conditions[i][unique_conditions[i] != 0][0])**2,3))

fig, ax1 = plt.subplots(1,1)

import matplotlib.colors as colors
negatives = estimated_pert.min()
positives = estimated_pert.max()


num_neg_colors = int(256 / (positives - negatives) * (-negatives))
num_pos_colors = 256 - num_neg_colors
cmap_BuRd = plt.cm.RdBu_r
colors_2neg_4pos = [cmap_BuRd(0.5*c/num_neg_colors) for c in range(num_neg_colors)] +\
                   [cmap_BuRd(1-0.5*c/num_pos_colors) for c in range(num_pos_colors)][::-1]
cmap_2neg_4pos = colors.LinearSegmentedColormap.from_list('cmap_2neg_4pos', colors_2neg_4pos, N=256)

im = ax1.imshow(estimated_pert[1:], cmap=cmap_2neg_4pos)
for i in [6.5, 13.5, 20.5]:
    ax1.axline((i, i), slope=0, alpha=0.4, c='k')
ax1.set_xticks(np.arange(len(my_gene_name)), my_gene_name, rotation=90)
ax1.set_yticks(np.arange(len(my_yticks)-1), my_yticks[1:])
ax1.set_title('Estimated perturbation effects')
ticks = np.append(np.arange(-60, 0., 10.), np.arange(0, 120, 20.))
fig.colorbar(im, ax=ax1, ticks=ticks)
fig.set_size_inches(12, 8)
fig.tight_layout()
plt.savefig('BSAPR_heatmap_SciPlex2_ZIP.png')
plt.close()


perturb_level, _, base_mean, logit_p, _ = parametric_model(my_conditioner, my_cell_info)
estimated_pert = logexpp1(F.sigmoid(logit_p) * perturb_level + base_mean).detach().cpu().numpy()
fig, axs = plt.subplots(1, 1)
axs.scatter(estimated_pert[my_observation != 0], my_observation[my_observation != 0].numpy(), alpha=0.1)
axs.axline((1, 1), slope=1, c='r', alpha=0.5, linestyle='--')
axs.set_xlabel('Averaged predicted Poisson rate')
axs.set_ylabel('Averaged observed counts')
axs.set_title('Prediction vs Observation, BSAPR ZIP')
fig.set_size_inches(10, 10)
fig.tight_layout()
plt.savefig('BSAPR_pred_vs_obs_SciPlex2_ZIP.png')
plt.close()








