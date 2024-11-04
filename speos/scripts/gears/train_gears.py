from gears import PertData, GEARS

# get data
print("initializing data")
pert_data = PertData('/mnt/storage/gears/')
print("Done")
# load dataset in paper: norman, adamson, dixit.
print("loading data")
pert_data.load(data_name = 'norman')
print("Done")

# specify data split
pert_data.prepare_split(split = 'simulation', seed = 1)
# get dataloader with batch size
pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)
print(pert_data.gene_names)
print(len(pert_data.gene_names))
print("initializing model")
# set up and train a model
gears_model = GEARS(pert_data, device='cuda:0')
#gears_model.model_initialize(hidden_size = 64)
print("Done")
print("starting training")
gears_model.train(epochs = 20)
print("Done")
# save/load model
gears_model.save_model('gears')
gears_model.load_pretrained('gears')

# predict
#print(gears_model.predict([['CBL', 'CNN1'], ['FEV']]))
#gears_model.GI_predict(['CBL', 'CNN1'], GI_genes_file=None)

print(gears_model.predict([["PARK7"]]))
