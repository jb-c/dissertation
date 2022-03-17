import sys,torch,torchcde,os
sys.path.append('../')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
device = torch.device("cuda")
from regression_cde import CDEModel
from experiments.load_data_into_tensors import return_X_y_train_and_test

#---------------------- Data Prep ----------------
X_train, X_test, y_train, y_test = return_X_y_train_and_test(lookback_window_in_days=365,test_frac=0.15,random_state=42)

# Prep data by turning it into a continuous path
if 'cubic' == 'cubic':
    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_train[:3]).to(device)
    test_coeffs  = torchcde.hermite_cubic_coefficients_with_backward_differences(X_test).to(device)


#----------- Model Definitions ----------------
cde = CDEModel(8, 32, 2, interpolation_method = 'cubic', sde = False,adjoint=False).to(device)
sde = CDEModel(8, 32, 2, interpolation_method = 'cubic', sde = True,adjoint=False,method='reversible_heun',dropout_p = 0.7).to(device)


y_cde = cde(train_coeffs,return_path=True)
y_sde = sde(train_coeffs, return_path=True, num_stochastic_samples=10)
print('Done')