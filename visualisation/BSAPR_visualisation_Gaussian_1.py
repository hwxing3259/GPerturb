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

fitted_vals = Gaussian_estimates(model=parametric_model, obs=my_observation.to(device), cond=my_conditioner.to(device),
                               cell_info=my_cell_info.to(device))




pert_mean_LUHMES = fitted_vals['pert_mean'].ravel() * 1.0
obs_LUHMES = fitted_vals['obs'].ravel() * 1.0
inclusion_LUHMES = fitted_vals['prob'] > threshold
normed_pert_effect_LUHMES = fitted_vals['pert_effect']/np.sqrt(fitted_vals['base_removed'].var(axis=0)[None, :])
normed_pert_removed_LUHMES = fitted_vals['base_removed']/np.sqrt(fitted_vals['base_removed'].var(axis=0)[None, :])
normed_pert_effect_LUHMES = normed_pert_effect_LUHMES[inclusion_LUHMES]
normed_pert_removed_LUHMES = normed_pert_removed_LUHMES[inclusion_LUHMES]
normed_error_LUHMES_BSAPR = (normed_pert_effect_LUHMES - normed_pert_removed_LUHMES)/(normed_pert_removed_LUHMES+1e-2)


GSFA_fit_LUHMES = (pandas.read_csv('LUHMES_GSFA_fit.csv', index_col=0).to_numpy())[inclusion_LUHMES]
GSFA_obs_LUHMES = (pandas.read_csv('LUHMES_scaled_gene_exp.csv').to_numpy())[inclusion_LUHMES]
normed_error_LUHMES_GSFA = (GSFA_fit_LUHMES - GSFA_obs_LUHMES)/(GSFA_obs_LUHMES+1e-2)

print('Corr_LUHMES_full = {}'.format(np.round(np.corrcoef(pert_mean_LUHMES, obs_LUHMES)[0, 1], 4)))
print('Corr_LUHMES_pert = {}'.format(np.round(np.corrcoef(normed_pert_effect_LUHMES, normed_pert_removed_LUHMES)[0, 1], 4)))
print('Corr_LUHMES_GSFA = {}'.format(np.round(np.corrcoef(GSFA_fit_LUHMES, GSFA_obs_LUHMES)[0, 1], 4)))

fig, axes = plt.subplots(2,3)  # plt.subplots(nrows=2, ncols=2)
axes[0, 0].scatter(pert_mean_LUHMES, obs_LUHMES, alpha=0.25)
axes[0, 0].axline((1, 1), slope=1, c='r', alpha=0.5, linestyle='--')
axes[0, 0].set_xlabel('Predicted perturbed responses')
axes[0, 0].set_ylabel('Observed responses')
axes[0, 0].set_title('LUHMES, Observed vs Predicted')
# axes[0, 0].text(20, -40, 'Corr = {}'.format(np.round(np.corrcoef(pert_mean_LUHMES,
#                                                                  obs_LUHMES)[0, 1], 4)))

axes[0, 1].scatter(normed_pert_effect_LUHMES,
                   normed_pert_removed_LUHMES, alpha=0.25)
axes[0, 1].axline((1, 1), slope=1, c='r', alpha=0.5, linestyle='--')
axes[0, 1].set_ylim(-6.5, 25)
axes[0, 1].set_xlabel('Predicted perturbations')
axes[0, 1].set_ylabel('Observed responses with cell-level estimates removed')
axes[0, 1].set_title('Predicted vs Observed, cell level estimates removed')
# axes[0, 1].text(0.75, -8, 'Corr = {}'.format(np.round(np.corrcoef(normed_pert_effect_LUHMES,
#                                                                   normed_pert_removed_LUHMES)[0, 1], 4)))
print('here1')
axes[0, 2].scatter(GSFA_fit_LUHMES, GSFA_obs_LUHMES, alpha=0.25)
axes[0, 2].axline((1, 1), slope=1, c='r', alpha=0.5, linestyle='--')
axes[0, 2].set_ylim(-6.5, 25)
axes[0, 2].set_xlabel('Predicted perturbations')
axes[0, 2].set_ylabel('Observed responses with cell-level estimates removed')
axes[0, 2].set_title('GSFA, Predicted vs Observed, cell level estimates removed')
# axes[0, 2].text(0.75, -8, 'Corr = {}'.format(np.round(np.corrcoef(GSFA_fit_LUHMES,
#                                                                   GSFA_obs_LUHMES)[0, 1], 4)))


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

fitted_vals = Gaussian_estimates(model=parametric_model, obs=my_observation_0.to(device), cond=my_conditioner_0.to(device),
                                 cell_info=my_cell_info_0.to(device))

pert_mean[stimulate_idx] = fitted_vals['pert_mean'] * 1.0
obs[stimulate_idx] = fitted_vals['obs'] * 1.0
pert_effect[stimulate_idx] = fitted_vals['pert_effect'] * 1.0
base_removed[stimulate_idx] = fitted_vals['base_removed'] * 1.0
prob[stimulate_idx] = fitted_vals['prob'] * 1.0





my_observation_0, my_cell_info_0, my_conditioner_0 = my_observation[~stimulate_idx], my_cell_info[~stimulate_idx], my_conditioner[~stimulate_idx]
N = my_observation_0.shape[0]
parametric_model = BSAPR_Gaussian(conditioner_dim=conditioner_dim, output_dim=output_dim, base_dim=cell_info_dim,
                                  data_size = N, hidden_node=hidden_node, hidden_layer_1=hidden_layer_1,
                                  hidden_layer_2=hidden_layer_2, tau=tau)
parametric_model = parametric_model.to(device)
parametric_model.load_state_dict(torch.load('BSAPR_TCells_Gaussian_cond1.pt'))

fitted_vals = Gaussian_estimates(model=parametric_model, obs=my_observation_0.to(device), cond=my_conditioner_0.to(device),
                                 cell_info=my_cell_info_0.to(device))

pert_mean[~stimulate_idx] = fitted_vals['pert_mean'] * 1.0
obs[~stimulate_idx] = fitted_vals['obs'] * 1.0
pert_effect[~stimulate_idx] = fitted_vals['pert_effect'] * 1.0
base_removed[~stimulate_idx] = fitted_vals['base_removed'] * 1.0
prob[~stimulate_idx] = fitted_vals['prob'] * 1.0





inclusion_TCells = prob > threshold
normed_pert_effect_TCells = (pert_effect/np.sqrt(base_removed.var(axis=0)[None, :]))[inclusion_TCells]
normed_pert_removed_TCells = (base_removed / np.sqrt(base_removed.var(axis=0)[None, :]))[inclusion_TCells]
normed_error_TCells_BSAPR = (normed_pert_effect_TCells - normed_pert_removed_TCells)/(normed_pert_removed_TCells+1e-2)

GSFA_fit = pandas.read_csv('TCell_GSFA_fit.csv', index_col=0).to_numpy()[inclusion_TCells]
GSFA_obs = pandas.read_csv('TCells_scaled_gene_exp.csv').to_numpy()[inclusion_TCells]
normed_error_TCells_GSFA = (GSFA_fit - GSFA_obs)/(GSFA_obs+1e-2)

print('Corr_TCells_full = {}'.format(np.round(np.corrcoef(pert_mean.ravel(), obs.ravel())[0, 1], 4)))
print('Corr_TCells_pert = {}'.format(np.round(np.corrcoef(normed_pert_effect_TCells, normed_pert_removed_TCells)[0, 1], 4)))
print('Corr_TCells_GSFA = {}'.format(np.round(np.corrcoef(GSFA_fit, GSFA_obs)[0, 1], 4)))

axes[1, 0].scatter(pert_mean.ravel(), obs.ravel(), alpha=0.15)
axes[1, 0].axline((1, 1), slope=1, c='r', alpha=0.5, linestyle='--')
axes[1, 0].set_xlabel('Predicted perturbed responses')
axes[1, 0].set_ylabel('Observed responses')
axes[1, 0].set_title('TCells, Predicted vs Observed')
# axes[1, 0].text(15, -27, 'Corr = {}'.format(np.round(np.corrcoef(pert_mean.ravel(),
#                                                                  obs.ravel())[0, 1], 4)))

axes[1, 1].scatter(normed_pert_effect_TCells,
                   normed_pert_removed_TCells,
                            alpha=0.25)
axes[1, 1].axline((1, 1), slope=1, c='r', alpha=0.5, linestyle='--')
axes[1, 1].set_ylim(-5, 20)
axes[1, 1].set_xlabel('Predicted perturbations')
axes[1, 1].set_ylabel('Observed responses with cell-level estimates removed')
axes[1, 1].set_title('Predicted vs Observed, cell level estimates removed')
# axes[1, 1].text(0.75, -10, 'Corr = {}'.format(np.round(np.corrcoef(normed_pert_effect_TCells,
#                                                                    normed_pert_removed_TCells)[0, 1], 4)))


axes[1, 2].scatter(GSFA_fit, GSFA_obs, alpha=0.15)
axes[1, 2].axline((1, 1), slope=1, c='r', alpha=0.5, linestyle='--')
axes[1, 2].set_ylim(-5, 20)
axes[1, 2].set_xlabel('Predicted perturbations')
axes[1, 2].set_ylabel('Observed responses with cell-level estimates removed')
axes[1, 2].set_title('GSFA, Predicted vs Observed, cell level estimates removed')
# axes[1, 2].text(0.75, -10, 'Corr = {}'.format(np.round(np.corrcoef(GSFA_fit,
#                                                                    GSFA_obs)[0, 1], 4)))

fig.set_size_inches(18, 12)
fig.tight_layout()
plt.savefig('BSAPR_vs_GSFA_2.png')
plt.close()
