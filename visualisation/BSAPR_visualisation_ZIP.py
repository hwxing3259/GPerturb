import pandas
import copy
from BSAPR_model import *
import matplotlib.colors as colors
device = 'cpu'
top = 50
threshold = 0.95
torch.manual_seed(3141592)



# load data:
my_conditioner = pd.read_csv('LUHMES_data/LUHMES_perturbation_mat.csv')
my_conditioner = my_conditioner.drop('Nontargeting', axis=1)  # TODO: or retaining it
cond_name = list(my_conditioner.columns)
my_conditioner = torch.tensor(my_conditioner.to_numpy() * 1.0, dtype=torch.float)


my_observation = pd.read_csv('LUHMES_data/LUHMES_counts_unfiltered_raw.csv', index_col=0)
print(my_observation.shape)
gene_name = list(my_observation.columns)
my_observation = torch.tensor(my_observation.to_numpy() * 1.0, dtype=torch.float)


gene_name_lookup = pd.read_csv('LUHMES_data/LUHMES_unfiltered_gene_lookup.csv', index_col=0)
gene_name_lookup = {gene_name_lookup.iloc[i].unfiltered_genes: gene_name_lookup.iloc[i].feature_name_lookup_unfiltered for i in range(gene_name_lookup.shape[0])}
gene_name = [gene_name_lookup[i] for i in gene_name]


my_cell_info = pd.read_csv('LUHMES_data/LUHMES_cell_information.csv', index_col=0)
my_cell_info = my_cell_info[['lib_size', 'batch', 'umi_count', 'percent_mt']]
my_cell_info.batch = pd.factorize(my_cell_info.batch)[0]
my_cell_info.umi_count = my_cell_info.umi_count/my_cell_info.lib_size
my_cell_info.lib_size = np.log(my_cell_info.lib_size)
cell_info_names = list(my_cell_info.columns)
my_cell_info = torch.tensor(my_cell_info.to_numpy() * 1.0, dtype=torch.float)


# design the training process:
output_dim = my_observation.shape[1]
sample_size = my_observation.shape[0]
hidden_node = 1000
hidden_layer_1 = 4
hidden_layer_2 = 4
conditioner_dim = my_conditioner.shape[1]
cell_info_dim = my_cell_info.shape[1]

lr_parametric = 5e-4  # 7e-4 works, 2e-3 kinda works
nu_1, nu_2, nu_3, nu_4 = torch.tensor(5.).to(device), torch.tensor(0.1).to(device), torch.tensor(3.).to(device), torch.tensor(0.1).to(device)
tau = torch.tensor(1.).to(device)

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

parametric_model = BSAPR_ZIP(conditioner_dim=conditioner_dim, output_dim=output_dim,
                             base_dim=cell_info_dim, data_size=sample_size,
                             hidden_node=hidden_node, hidden_layer_1=hidden_layer_1,
                             hidden_layer_2=hidden_layer_2, tau=tau)
parametric_model = parametric_model.to(device)
parametric_model.load_state_dict(torch.load('BSAPR_LUHMES_ZIP_ref.pt'))


unique_conditions = torch.unique(my_conditioner, dim=0).to(device)
perturb_level, _, _, logit_p, _ = parametric_model(unique_conditions, None)
estimated_inclusion_prob = F.sigmoid(logit_p).detach().cpu().numpy()
estimated_inclusion = estimated_inclusion_prob > threshold
my_gene_name = np.array(gene_name)[estimated_inclusion.sum(axis=0) > 0]
estimated_inclusion_prob = estimated_inclusion_prob[:, estimated_inclusion.mean(axis=0) > 0]
estimated_pert = perturb_level.detach().cpu().numpy() * estimated_inclusion
estimated_pert = estimated_pert[:, estimated_inclusion.mean(axis=0) > 0]
top_gene = np.argsort((estimated_pert**2).sum(axis=0))[::-1][:top]


unique_conditions = unique_conditions.cpu().numpy()
my_yticks = ['' for _ in range(unique_conditions.shape[0])]
for i in range(unique_conditions.shape[0]):
    if np.all(unique_conditions[i] == 0):
        my_yticks[i] = 'Non Targeting'
    else:
        my_yticks[i] = np.array(cond_name)[unique_conditions[i] == 1][0]


estimated_pert_LUHMES = estimated_pert[1:, top_gene]*1.0
my_gene_name_LUHMES = copy.copy(my_gene_name[top_gene])
my_yticks_LUHMES = copy.copy(my_yticks[1:])



fig2, axes2 = plt.subplots(1, 1)
# plt.set_cmap('RdBu')
negatives = estimated_pert_LUHMES.min()
positives = estimated_pert_LUHMES.max()

# bounds_min = np.linspace(negatives, 0, 129)
# bounds_max = np.linspace(0, positives, 129)[1:]
#     # the zero is only needed once
#     # in total there will be 257 bounds, so 256 bins
# bounds = np.concatenate((bounds_min, bounds_max), axis=None)
# norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
#
# x, y = np.meshgrid(np.arange(estimated_pert_LUHMES.shape[1]), np.arange(estimated_pert_LUHMES.shape[0]))
# im = axes2[0].pcolormesh(x,y,estimated_pert_LUHMES, norm=norm, cmap="RdBu_r")

num_neg_colors = int(256 / (positives - negatives) * (-negatives))
num_pos_colors = 256 - num_neg_colors
cmap_BuRd = plt.cm.RdBu_r
colors_2neg_4pos = [cmap_BuRd(0.5*c/num_neg_colors) for c in range(num_neg_colors)] +\
                   [cmap_BuRd(1-0.5*c/num_pos_colors) for c in range(num_pos_colors)][::-1]
cmap_2neg_4pos = colors.LinearSegmentedColormap.from_list('cmap_2neg_4pos', colors_2neg_4pos, N=256)

im = axes2.imshow(estimated_pert_LUHMES, cmap=cmap_2neg_4pos)
# im = axes2.imshow(estimated_pert_LUHMES, norm=norm, cmap='RdBu_r')
axes2.set_xticks(np.arange(len(my_gene_name_LUHMES)), my_gene_name_LUHMES, rotation=90)
axes2.set_yticks(np.arange(len(my_yticks_LUHMES)), my_yticks_LUHMES)
axes2.set_ylabel('Perturbation names')
axes2.set_xlabel('Gene names')
axes2.set_title('Estimated perturbation effects')
# axes2.set_aspect('equal')
# ticks = np.append(np.arange(-6, 0, 1), np.arange(0, 60.001, 10))
# fig2.colorbar(im, ax=axes2, ticks=ticks)
fig2.colorbar(im, ax=axes2)
fig2.set_size_inches(12, 6)
fig2.tight_layout()
plt.savefig('LUHMES_BSAPR_ZIP_heatmap_2.png')
plt.close()



predicted_mu_mean, predicted_mu_log_var, predicted_base_mean, logit_p, logit_p_log_var = parametric_model(my_conditioner.to(device), my_cell_info.to(device))
estimated_base_mean = predicted_base_mean.detach().cpu().numpy()  # * zeros[testing_idx].numpy()
estimated_perturbed_mean = (F.sigmoid(logit_p) * predicted_mu_mean).detach().cpu().numpy()
estimated_perturbed_rate = (logexpp1(torch.tensor(estimated_base_mean + estimated_perturbed_mean))).numpy()

fig2, axes2 = plt.subplots(1, 1)
my_observation = my_observation.numpy()
axes2.scatter(estimated_perturbed_rate[my_observation != 0], my_observation[my_observation != 0], alpha=0.15)
axes2.axline((1, 1), slope=1, c='r', linestyle='--', alpha=0.5)
axes2.set_ylabel('Observed counts')
axes2.set_xlabel('Predicted Poisson rate')
axes2.set_title('Predicted counts vs observations')
fig2.set_size_inches(12, 12)
fig2.tight_layout()
plt.savefig('LUHMES_BSAPR_ZIP_obs_vs_pred_2.png')
plt.close()












my_conditioner = pd.read_csv('TCells_data/TCells_perturbation_mat.csv')
my_conditioner = my_conditioner.drop('NonTarget', axis=1)  # TODO: or retaining it
cond_name = list(my_conditioner.columns)
my_conditioner = torch.tensor(my_conditioner.to_numpy() * 1.0, dtype=torch.float)

my_observation = pd.read_csv('TCells_data/TCells_count_unfiltered_raw.csv', index_col=0)
print(my_observation.shape)
gene_name = list(my_observation.columns)
my_observation = torch.tensor(my_observation.to_numpy() * 1.0, dtype=torch.float)

gene_name_lookup = pd.read_csv('TCells_data/TCells_unfiltered_gene_lookup.csv', index_col=0)
gene_name_lookup = {gene_name_lookup.iloc[i].unfiltered_genes: gene_name_lookup.iloc[i].feature_name_lookup_unfiltered for i in range(gene_name_lookup.shape[0])}
gene_name = [gene_name_lookup[i] for i in gene_name]

my_cell_info = pd.read_csv('TCells_data/TCells_cell_information.csv', index_col=0)
my_cell_info = my_cell_info[['lib_size', 'umi_count', 'percent_mt', 'doner', 'stimulated']]
my_cell_info.umi_count = my_cell_info.umi_count/my_cell_info.lib_size
my_cell_info.lib_size = np.log(my_cell_info.lib_size)
cell_info_names = list(my_cell_info.columns)
my_cell_info = torch.tensor(my_cell_info.to_numpy() * 1.0, dtype=torch.float)

stimulate_idx = my_cell_info[:, 4] == 0.
my_cell_info = my_cell_info[:, :4]

# design the training process:
start = time.time()
output_dim = my_observation.shape[1]
hidden_node =1000
hidden_layer_1 = 4
hidden_layer_2 = 4
conditioner_dim = my_conditioner.shape[1]
cell_info_dim = my_cell_info.shape[1]

lr_parametric = 3e-4  # 7e-4 works, 2e-3 kinda works
nu_1, nu_2, nu_3, nu_4 = torch.tensor(3.).to(device), torch.tensor(0.1).to(device), torch.tensor(3.).to(device), torch.tensor(0.1).to(device)
tau = torch.tensor(1.).to(device)
KL_factor = torch.tensor(1.).to(device)


torch.manual_seed(314159)
np.random.seed(314159)
my_observation_0, my_cell_info_0, my_conditioner_0 = my_observation[stimulate_idx], my_cell_info[stimulate_idx], my_conditioner[stimulate_idx]
my_dataset = MyDataSet(observation=my_observation_0, conditioner=my_conditioner_0, cell_info=my_cell_info_0)
N = my_observation_0.shape[0]
testing_idx = set(np.random.choice(a=range(my_observation_0.shape[0]), size=my_observation_0.shape[0]//8, replace=False))
training_idx = list(set(range(my_observation_0.shape[0])) - testing_idx)
testing_idx = list(testing_idx)
training_idx_sampler = torch.utils.data.SubsetRandomSampler(training_idx)
training_loader = torch.utils.data.DataLoader(my_dataset, batch_size=256, sampler=training_idx_sampler)
test_observation = my_observation_0[testing_idx].to(device)
test_conditioner = my_conditioner_0[testing_idx].to(device)
test_cell_info = my_cell_info_0[testing_idx].to(device)

parametric_model = BSAPR_ZIP(conditioner_dim=conditioner_dim, output_dim=output_dim,
                                    base_dim=cell_info_dim, data_size=N,
                                    hidden_node=hidden_node, hidden_layer_1=hidden_layer_1,
                                    hidden_layer_2=hidden_layer_2, tau=tau)
parametric_model = parametric_model.to(device)

parametric_model.load_state_dict(torch.load('BSAPR_TCells_ZIP_cond0.pt'))

predicted_mu_mean, predicted_mu_log_var, predicted_base_mean, logit_p, logit_p_log_var = parametric_model(test_conditioner, test_cell_info)
estimated_base_mean = predicted_base_mean.detach().cpu().numpy()  # * zeros[testing_idx].numpy()
estimated_perturbed_mean = (F.sigmoid(logit_p) * predicted_mu_mean).detach().cpu().numpy()
estimated_perturbed_rate = (logexpp1(torch.tensor(estimated_base_mean + estimated_perturbed_mean))).numpy()
est_0 = {'non_zero_pred': estimated_perturbed_rate[test_observation.cpu() != 0], 'non_zero_obs': test_observation.cpu()[test_observation.cpu() != 0].numpy()}

threshold = 0.95
top = 50
unique_conditions = torch.unique(my_conditioner_0, dim=0).to(device)
perturb_level, _, _, logit_p, _ = parametric_model(unique_conditions, None)
estimated_inclusion_prob = F.sigmoid(logit_p).detach().cpu().numpy()
estimated_inclusion = estimated_inclusion_prob > threshold
my_gene_name = np.array(gene_name)[estimated_inclusion.sum(axis=0) > 0]
estimated_inclusion_prob = estimated_inclusion_prob[:, estimated_inclusion.mean(axis=0) > 0]
estimated_pert = perturb_level.detach().cpu().numpy() * estimated_inclusion
estimated_pert = estimated_pert[:, estimated_inclusion.mean(axis=0) > 0]
top_gene = np.argsort((estimated_pert**2).sum(axis=0))[::-1][:top]
estimated_pert = estimated_pert[:, top_gene]

unique_conditions = unique_conditions.cpu().numpy()
my_yticks = ['' for _ in range(unique_conditions.shape[0])]
for i in range(unique_conditions.shape[0]):
    if np.all(unique_conditions[i] == 0):
        my_yticks[i] = 'Non Targeting'
    else:
        my_yticks[i] = np.array(cond_name)[unique_conditions[i] == 1][0] + '_N'


estimated_pert_0 = estimated_pert[1:, :]*1.0
my_gene_name_0 = copy.copy(my_gene_name[top_gene])
my_yticks_0 = copy.copy(my_yticks[1:])




my_observation_1, my_cell_info_1, my_conditioner_1 = my_observation[~stimulate_idx], my_cell_info[~stimulate_idx], my_conditioner[~stimulate_idx]
my_dataset = MyDataSet(observation=my_observation_1, conditioner=my_conditioner_1, cell_info=my_cell_info_1)
N = my_observation_1.shape[0]
testing_idx = set(np.random.choice(a=range(my_observation_1.shape[0]), size=my_observation_1.shape[0]//8, replace=False))
training_idx = list(set(range(my_observation_1.shape[0])) - testing_idx)
testing_idx = list(testing_idx)
training_idx_sampler = torch.utils.data.SubsetRandomSampler(training_idx)
training_loader = torch.utils.data.DataLoader(my_dataset, batch_size=256, sampler=training_idx_sampler)
test_observation = my_observation_1[testing_idx].to(device)
test_conditioner = my_conditioner_1[testing_idx].to(device)
test_cell_info = my_cell_info_1[testing_idx].to(device)

parametric_model = BSAPR_ZIP(conditioner_dim=conditioner_dim, output_dim=output_dim,
                                    base_dim=cell_info_dim, data_size=N,
                                    hidden_node=hidden_node, hidden_layer_1=hidden_layer_1,
                                    hidden_layer_2=hidden_layer_2, tau=tau)
parametric_model = parametric_model.to(device)


parametric_model.load_state_dict(torch.load('BSAPR_TCells_ZIP_cond1.pt'))

predicted_mu_mean, _, predicted_base_mean, logit_p, logit_p_log_var = parametric_model(test_conditioner, test_cell_info)
estimated_base_mean = predicted_base_mean.detach().cpu().numpy()  # * zeros[testing_idx].numpy()
estimated_perturbed_mean = (F.sigmoid(logit_p) * predicted_mu_mean).detach().cpu().numpy()
estimated_perturbed_rate = (logexpp1(torch.tensor(estimated_base_mean + estimated_perturbed_mean))).numpy()
est_1 = {'non_zero_pred': estimated_perturbed_rate[test_observation.cpu() != 0], 'non_zero_obs': test_observation.cpu()[test_observation.cpu() != 0].numpy()}


unique_conditions = torch.unique(my_conditioner_1, dim=0).to(device)
perturb_level, _, _, logit_p, _ = parametric_model(unique_conditions, None)
estimated_inclusion_prob = F.sigmoid(logit_p).detach().cpu().numpy()
estimated_inclusion = estimated_inclusion_prob > threshold
my_gene_name = np.array(gene_name)[estimated_inclusion.sum(axis=0) > 0]
estimated_inclusion_prob = estimated_inclusion_prob[:, estimated_inclusion.mean(axis=0) > 0]
estimated_pert = perturb_level.detach().cpu().numpy() * estimated_inclusion
estimated_pert = estimated_pert[:, estimated_inclusion.mean(axis=0) > 0]
top_gene = np.argsort((estimated_pert**2).sum(axis=0))[::-1][:top]
estimated_pert = estimated_pert[:, top_gene]

unique_conditions = unique_conditions.cpu().numpy()
my_yticks = ['' for _ in range(unique_conditions.shape[0])]
for i in range(unique_conditions.shape[0]):
    if np.all(unique_conditions[i] == 0):
        my_yticks[i] = 'Non Targeting'
    else:
        my_yticks[i] = np.array(cond_name)[unique_conditions[i] == 1][0] + '_S'


estimated_pert_1 = estimated_pert[1:, :]*1.0
my_gene_name_1 = copy.copy(my_gene_name[top_gene])
my_yticks_1 = copy.copy(my_yticks[1:])




fig2, axes2 = plt.subplots(1, 1)
my_observation = my_observation.numpy()
axes2.scatter(np.append(est_0['non_zero_pred'],est_1['non_zero_pred']), np.append(est_0['non_zero_obs'],est_1['non_zero_obs']), alpha=0.15)
axes2.axline((1, 1), slope=1, c='r', linestyle='--', alpha=0.5)
axes2.set_ylabel('Observed counts')
axes2.set_xlabel('Predicted Poisson rate')
axes2.set_title('Predicted counts vs observations')
fig2.set_size_inches(12, 12)
fig2.tight_layout()
plt.savefig('TCells_BSAPR_ZIP_obs_vs_pred_2.png')
plt.close()


fig2, axes2 = plt.subplots(1, 1)
# plt.set_cmap('RdBu')
estimated_pert_0 = pd.DataFrame(estimated_pert_0)
estimated_pert_0.columns = my_gene_name_0
estimated_pert_0.index = my_yticks_0

estimated_pert_1 = pd.DataFrame(estimated_pert_1)
estimated_pert_1.columns = my_gene_name_1
estimated_pert_1.index = my_yticks_1

estimated_pert = pd.concat([estimated_pert_0, estimated_pert_1], axis=0)
estimated_pert.fillna(0, inplace=True)
estimated_pert = estimated_pert.sort_index()
my_gene_name = np.array(list(estimated_pert.columns))
my_yticks = list(estimated_pert.index)
estimated_pert = estimated_pert.to_numpy()
sort_gene = np.argsort((estimated_pert**2).sum(axis=0))[::-1]
estimated_pert = estimated_pert[:, sort_gene]
my_gene_name = my_gene_name[sort_gene]


negatives = estimated_pert.min()
positives = estimated_pert.max()

# bounds_min = np.linspace(negatives, 0, 129)
# bounds_max = np.linspace(0, positives, 129)[1:]
#     # the zero is only needed once
#     # in total there will be 257 bounds, so 256 bins
# bounds = np.concatenate((bounds_min, bounds_max), axis=None)
# norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
#
# x, y = np.meshgrid(np.arange(estimated_pert_LUHMES.shape[1]), np.arange(estimated_pert_LUHMES.shape[0]))
# im = axes2[0].pcolormesh(x,y,estimated_pert_LUHMES, norm=norm, cmap="RdBu_r")

num_neg_colors = int(256 / (positives - negatives) * (-negatives))
num_pos_colors = 256 - num_neg_colors
cmap_BuRd = plt.cm.RdBu_r
colors_2neg_4pos = [cmap_BuRd(0.5*c/num_neg_colors) for c in range(num_neg_colors)] +\
                   [cmap_BuRd(1-0.5*c/num_pos_colors) for c in range(num_pos_colors)][::-1]
cmap_2neg_4pos = colors.LinearSegmentedColormap.from_list('cmap_2neg_4pos', colors_2neg_4pos, N=256)

im = axes2.imshow(estimated_pert, cmap=cmap_2neg_4pos)
# im = axes2.imshow(estimated_pert, norm=norm, cmap='RdBu_r')
axes2.set_xticks(np.arange(len(my_gene_name)), my_gene_name, rotation=90)
axes2.set_yticks(np.arange(len(my_yticks)), my_yticks)
axes2.set_ylabel('Perturbation names')
axes2.set_xlabel('Gene names')
axes2.set_title('Estimated perturbation effects')
# ticks = np.append(np.arange(-8, 0, 2.), np.arange(0, 10.001, 2.5))
# fig2.colorbar(im, ax=axes2, ticks=ticks)
fig2.colorbar(im, ax=axes2)
fig2.set_size_inches(9, 9)
fig2.tight_layout()
plt.savefig('BSAPR_heatmap_TCells_ZIP_merged.png')
plt.close()














