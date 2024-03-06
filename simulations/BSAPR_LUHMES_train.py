from BSAPR_model import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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

lr_parametric = 1e-3  # 7e-4 works, 2e-3 kinda works  # 1500, 1e-3, 0.005, 40, 0.37
nu_1, nu_2, nu_3, nu_4, nu_5, nu_6 = torch.tensor([1., 0.1, 1., 0.1, 1., 0.1]).to(device)
tau = torch.tensor(1.0).to(device)

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

optimizer = torch.optim.Adam(parametric_model.parameters(), lr=lr_parametric)


# training starts here
epoch = 300
training_loss = torch.zeros(epoch)
testing_loss = torch.zeros(epoch)
for EPOCH in range(epoch):
    curr_training_loss = 0.
    parametric_model.train()
    for i, (obs, cond, cell_info) in enumerate(training_loader):
        obs = obs.to(device)
        cond = cond.to(device)
        cell_info = cell_info.to(device)
        loss = parametric_model.normal_loss(observation=obs, conditioner=cond, cell_info=cell_info,
                                            nu_1=nu_1, nu_2=nu_2, nu_3=nu_3, nu_4=nu_4, nu_5=nu_5, nu_6=nu_6)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        curr_training_loss += loss.detach().cpu().item() * obs.shape[0]
    training_loss[EPOCH] = curr_training_loss/len(training_idx)
    parametric_model.eval()
    with torch.no_grad():
        testing_loss[EPOCH] = parametric_model.normal_loss(observation=test_observation, conditioner=test_conditioner,
                                                           cell_info=test_cell_info, nu_1=nu_1, nu_2=nu_2,
                                                           nu_3=nu_3, nu_4=nu_4,nu_5=nu_5, nu_6=nu_6, test=True).detach().cpu().item()
        print('EPOCH={}, test_error={}'.format(EPOCH, testing_loss[EPOCH]))
    print('EPOCH={}, training_error={}'.format(EPOCH, training_loss[EPOCH]))
    # sch.step()
end = time.time()
print(end-start)


torch.save(parametric_model.state_dict(), 'BSAPR_LUHMES_Gaussian_ref.pt')
np.savetxt(fname='BSAPR_LUHMES_Gaussian_training_loss_ref.txt', X=training_loss.numpy())
np.savetxt(fname='BSAPR_LUHMES_Gaussian_testing_loss_ref.txt', X=testing_loss.numpy())

my_plot_Gaussian(model=parametric_model, obs=test_observation, cond=test_conditioner,
                 cell_info=test_cell_info, full_cond=my_conditioner,
                 gene_name=gene_name, cond_name=cond_name, x=-20, y=30,
                 fig1='BSAPR_LUHMES_Gaussian_fig1_ref.png',
                 fig2='BSAPR_LUHMES_Gaussian_fig2_ref.png')






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
start = time.time()
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

parametric_model = BSAPR_ZIP(conditioner_dim=conditioner_dim, output_dim=output_dim,
                             base_dim=cell_info_dim, data_size=sample_size,
                             hidden_node=hidden_node, hidden_layer_1=hidden_layer_1,
                             hidden_layer_2=hidden_layer_2, tau=tau)
parametric_model = parametric_model.to(device)

optimizer = torch.optim.Adam(parametric_model.parameters(), lr=lr_parametric)


# training starts here
epoch = 300
training_loss = torch.zeros(epoch)
testing_loss = torch.zeros(epoch)
for EPOCH in range(epoch):
    curr_training_loss = 0.
    for i, (obs, cond, cell_info) in enumerate(training_loader):
        obs = obs.to(device)
        cond = cond.to(device)
        cell_info = cell_info.to(device)
        loss = parametric_model.zip_loss(observation=obs, conditioner=cond, cell_info=cell_info,
                                         nu_1=nu_1, nu_2=nu_2, nu_3=nu_3, nu_4=nu_4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        curr_training_loss += loss.detach().cpu().item() * obs.shape[0]
    training_loss[EPOCH] = curr_training_loss/len(training_idx)
    with torch.no_grad():
        testing_loss[EPOCH] = parametric_model.zip_loss(observation=test_observation,
                                                        conditioner=test_conditioner,
                                                        cell_info=test_cell_info,
                                                        nu_1=nu_1, nu_2=nu_2, nu_3=nu_3, nu_4=nu_4,
                                                        test=True).detach().cpu().item()
        print('EPOCH={}, test_error={}'.format(EPOCH, testing_loss[EPOCH]))
    print('EPOCH={}, training_error={}'.format(EPOCH, training_loss[EPOCH]))
    # sch.step()
end = time.time()
print(end-start)

torch.save(parametric_model.state_dict(), 'BSAPR_LUHMES_ZIP_ref.pt')
np.savetxt(fname='BSAPR_LUHMES_ZIP_training_loss_ref.txt', X=training_loss.numpy())
np.savetxt(fname='BSAPR_LUHMES_ZIP_testing_loss_ref.txt', X=testing_loss.numpy())

my_plot_ZIP(model=parametric_model, obs=test_observation, cond=test_conditioner,
            cell_info=test_cell_info, full_cond=my_conditioner,
            gene_name=gene_name, cond_name=cond_name, x=200, y=1500,
            fig1='BSAPR_LUHMES_ZIP_fig1_ref.png',
            fig2='BSAPR_LUHMES_ZIP_fig2_ref.png')





