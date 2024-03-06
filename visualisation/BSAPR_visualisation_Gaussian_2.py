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

my_observation = pd.read_csv('LUHMES_data/LUHMES_dev_filtered_raw.csv')
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
start = time.time()
output_dim = my_observation.shape[1]
sample_size = my_observation.shape[0]
hidden_node = 1000  # or 1000
hidden_layer_1 = 4
hidden_layer_2 = 4
conditioner_dim = my_conditioner.shape[1]
cell_info_dim = my_cell_info.shape[1]
tau = torch.tensor(1.).to(device)


torch.manual_seed(314159)

parametric_model = BSAPR_Gaussian(conditioner_dim=conditioner_dim, output_dim=output_dim, base_dim=cell_info_dim,
                               data_size=sample_size, hidden_node=hidden_node, hidden_layer_1=hidden_layer_1,
                               hidden_layer_2=hidden_layer_2, tau=tau)
parametric_model = parametric_model.to(device)
parametric_model.load_state_dict(torch.load('BSAPR_LUHMES_Gaussian_ref.pt'))


unique_conditions = torch.unique(my_conditioner, dim=0)
perturb_level, _, _, _, logit_p, _, _ = parametric_model(unique_conditions, None)
estimated_inclusion_prob = F.sigmoid(logit_p).detach().cpu().numpy()
estimated_inclusion = estimated_inclusion_prob > threshold
my_gene_name = np.array(gene_name)[estimated_inclusion.sum(axis=0) > 0]
estimated_inclusion_prob = estimated_inclusion_prob[:, estimated_inclusion.mean(axis=0) > 0]
estimated_pert = perturb_level.detach().cpu().numpy() * estimated_inclusion
estimated_pert = estimated_pert[:, estimated_inclusion.mean(axis=0) > 0]
top_gene = np.argsort((estimated_pert**2).sum(axis=0))[::-1][:top]

unique_conditions = unique_conditions.numpy()
my_yticks = ['' for _ in range(unique_conditions.shape[0])]
ref_id = 0
for i in range(unique_conditions.shape[0]):
    if np.all(unique_conditions[i] == 0):
        my_yticks[i] = 'Non Targeting'
        ref_id = i
    else:
        my_yticks[i] = np.array(cond_name)[unique_conditions[i] == 1][0]

estimated_pert_LUHMES = estimated_pert[1:, top_gene]*1.0
my_gene_name_LUHMES = copy.copy(my_gene_name[top_gene])
my_yticks_LUHMES = copy.copy(my_yticks[1:])




fig2, axes2 = plt.subplots(1, 1)
# plt.set_cmap('RdBu')
negatives = estimated_pert_LUHMES.min() - 0.1
positives = estimated_pert_LUHMES.max() + 0.1

bounds_min = np.linspace(negatives, 0, 129)
bounds_max = np.linspace(0, positives, 129)[1:]
    # the zero is only needed once
    # in total there will be 257 bounds, so 256 bins
bounds = np.concatenate((bounds_min, bounds_max), axis=None)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
#
# x, y = np.meshgrid(np.arange(estimated_pert_LUHMES.shape[1]), np.arange(estimated_pert_LUHMES.shape[0]))
# im = axes2[0].pcolormesh(x,y,estimated_pert_LUHMES, norm=norm, cmap="RdBu_r")


negatives = estimated_pert.min()
positives = estimated_pert.max()
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
axes2.set_aspect('equal')
# ticks = np.append(np.arange(-1.5, 0, 0.5), np.arange(0, 5.101, 1.7))
# fig2.colorbar(im, ax=axes2, ticks=ticks)
fig2.colorbar(im, ax=axes2)
fig2.set_size_inches(12, 6)
fig2.tight_layout()
plt.savefig('LUHMES_BSAPR_heatmap_2.png')
plt.close()












# generate synthetic data
device = 'cpu'
torch.manual_seed(3141592)
# load data:
my_conditioner = pd.read_csv('TCells_data/TCells_perturbation_mat.csv')
my_conditioner = my_conditioner.drop('NonTarget', axis=1)  # TODO: or retaining it
cond_name = list(my_conditioner.columns)
my_conditioner = torch.tensor(my_conditioner.to_numpy() * 1.0, dtype=torch.float)

my_observation = pd.read_csv('TCells_data/TCells_dev_filtered_raw.csv')
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
hidden_node = 1000
hidden_layer_1 = 4
hidden_layer_2 = 4
conditioner_dim = my_conditioner.shape[1]
cell_info_dim = my_cell_info.shape[1]
tau = torch.tensor(1.).to(device)

pert_mean = np.zeros((my_observation.shape[0], output_dim))
obs = np.zeros((my_observation.shape[0], output_dim))
pert_effect = np.zeros((my_observation.shape[0], output_dim))
base_removed = np.zeros((my_observation.shape[0], output_dim))
prob = np.zeros((my_observation.shape[0], output_dim))


my_observation_0, my_cell_info_0, my_conditioner_0 = my_observation[stimulate_idx], my_cell_info[stimulate_idx], my_conditioner[stimulate_idx]
N = my_observation_0.shape[0]
parametric_model = BSAPR_Gaussian(conditioner_dim=conditioner_dim, output_dim=output_dim, base_dim=cell_info_dim,
                                  data_size = N, hidden_node=hidden_node, hidden_layer_1=hidden_layer_1,
                                  hidden_layer_2=hidden_layer_2, tau=tau)
parametric_model = parametric_model.to(device)
parametric_model.load_state_dict(torch.load('BSAPR_TCells_Gaussian_cond0.pt'))


unique_conditions = torch.unique(my_conditioner_0, dim=0)
perturb_level, _, _, _, logit_p, _, _ = parametric_model(unique_conditions, None)
estimated_inclusion_prob = F.sigmoid(logit_p).detach().cpu().numpy()
estimated_inclusion = estimated_inclusion_prob > threshold
my_gene_name = np.array(gene_name)[estimated_inclusion.sum(axis=0) > 0]
estimated_inclusion_prob = estimated_inclusion_prob[:, estimated_inclusion.mean(axis=0) > 0]
estimated_pert = perturb_level.detach().cpu().numpy() * estimated_inclusion
estimated_pert = estimated_pert[:, estimated_inclusion.mean(axis=0) > 0]
top_gene = np.argsort((estimated_pert**2).sum(axis=0))[::-1][:top]
estimated_pert = estimated_pert[:, top_gene]

unique_conditions = unique_conditions.numpy()
my_yticks = ['' for _ in range(unique_conditions.shape[0])]
for i in range(unique_conditions.shape[0]):
    if np.all(unique_conditions[i] == 0):
        my_yticks[i] = 'Non Targeting'
    else:
        my_yticks[i] = np.array(cond_name)[unique_conditions[i] == 1][0] + '_N'

estimated_pert_0 = estimated_pert[1:, :]*1.0
my_gene_name_0 = copy.copy(my_gene_name[top_gene])
my_yticks_0 = copy.copy(my_yticks[1:])




my_observation_0, my_cell_info_0, my_conditioner_0 = my_observation[~stimulate_idx], my_cell_info[~stimulate_idx], my_conditioner[~stimulate_idx]
N = my_observation_0.shape[0]
parametric_model = BSAPR_Gaussian(conditioner_dim=conditioner_dim, output_dim=output_dim, base_dim=cell_info_dim,
                                  data_size = N, hidden_node=hidden_node, hidden_layer_1=hidden_layer_1,
                                  hidden_layer_2=hidden_layer_2, tau=tau)
parametric_model = parametric_model.to(device)
parametric_model.load_state_dict(torch.load('BSAPR_TCells_Gaussian_cond1.pt'))


unique_conditions = torch.unique(my_conditioner_0, dim=0)
perturb_level, _, _, _, logit_p, _, _ = parametric_model(unique_conditions, None)
estimated_inclusion_prob = F.sigmoid(logit_p).detach().cpu().numpy()
estimated_inclusion = estimated_inclusion_prob > threshold
my_gene_name = np.array(gene_name)[estimated_inclusion.sum(axis=0) > 0]
estimated_inclusion_prob = estimated_inclusion_prob[:, estimated_inclusion.mean(axis=0) > 0]
estimated_pert = perturb_level.detach().cpu().numpy() * estimated_inclusion
estimated_pert = estimated_pert[:, estimated_inclusion.mean(axis=0) > 0]
top_gene = np.argsort((estimated_pert**2).sum(axis=0))[::-1][:top]
estimated_pert = estimated_pert[:, top_gene]

unique_conditions = unique_conditions.numpy()
my_yticks = ['' for _ in range(unique_conditions.shape[0])]
for i in range(unique_conditions.shape[0]):
    if np.all(unique_conditions[i] == 0):
        my_yticks[i] = 'Non Targeting'
    else:
        my_yticks[i] = np.array(cond_name)[unique_conditions[i] == 1][0] + '_S'

estimated_pert_1 = estimated_pert[1:, :]*1.0
my_gene_name_1 = copy.copy(my_gene_name[top_gene])
my_yticks_1 = copy.copy(my_yticks[1:])



# plt.set_cmap('RdBu')
estimated_pert_0 = pandas.DataFrame(estimated_pert_0)
estimated_pert_0.columns = my_gene_name_0
estimated_pert_0.index = my_yticks_0

estimated_pert_1 = pandas.DataFrame(estimated_pert_1)
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
num_neg_colors = int(256 / (positives - negatives) * (-negatives))
num_pos_colors = 256 - num_neg_colors
cmap_BuRd = plt.cm.RdBu_r
colors_2neg_4pos = [cmap_BuRd(0.5*c/num_neg_colors) for c in range(num_neg_colors)] +\
                   [cmap_BuRd(1-0.5*c/num_pos_colors) for c in range(num_pos_colors)][::-1]
cmap_2neg_4pos = colors.LinearSegmentedColormap.from_list('cmap_2neg_4pos', colors_2neg_4pos, N=256)


fig2, axes2 = plt.subplots(1, 1)
# im = axes2.imshow(estimated_pert, norm=norm, cmap='RdBu_r')
im = axes2.imshow(estimated_pert, cmap=cmap_2neg_4pos)
axes2.set_xticks(np.arange(len(my_gene_name)), my_gene_name, rotation=90)
axes2.set_yticks(np.arange(len(my_yticks)), my_yticks)
axes2.set_ylabel('Perturbation names')
axes2.set_xlabel('Gene names')
axes2.set_title('Estimated perturbation effects')
# axes2.set_aspect('equal')
# ticks = np.append(np.arange(-.05, 0, 0.05), np.arange(0, 1.5, 0.75))
# fig2.colorbar(im, ax=axes2, ticks=ticks)
fig2.colorbar(im, ax=axes2)
fig2.set_size_inches(12, 9)
fig2.tight_layout()
plt.savefig('BSAPR_heatmap_TCells_merged_2.png')
plt.close()









