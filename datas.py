import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.data as data_utils
import numpy as np
import os
import scipy.io

def load_data(params):
  # Load data
  torch.manual_seed(params['random_seed'])
  np.random.seed(params['random_seed'])
  if  params['dataset'] == 'mnist':
    loader = load_mnist
  elif params['dataset'] == 'omniglot':
    loader = load_omniglot
  data_loaders = loader(params['batch_size'],binarize=params['binarize'],split_valid=params['split_valid'])

  return data_loaders

def get_balanced_dataset(ds,size = 100, num_classes = 10):
    """ ds: mnist data set
        size: should be divisible by 10 (num of classes)
        Returns: balanced dataset with size/num_classes samples for each class"""
    assert (size % num_classes == 0), "size is not divisible by num_classes"
    dl = torch.utils.data.DataLoader(dataset=ds,
                                     batch_size=1,
                                     shuffle=True)
    single_size = ds[0][0].size()
    tensor_im = torch.zeros(single_size)
    all_size = [tensor_im for _ in range(size)]
    tensor_im = torch.stack(all_size)
    tensor_la = torch.zeros(size).long()
    idx = 0
    balance = [0 for _ in range(num_classes)]
    print ("collecting a balanced sub-set of {} samples".format(size),)
    for im,la in dl:
        if balance[la[0]] < size/num_classes:
            balance[la[0]] += 1
            tensor_im[idx] = im
            tensor_la[idx] = la[0]
            idx += 1

        if idx == size:
            break
    print ()
    return torch.utils.data.TensorDataset(tensor_im,tensor_la)

def get_pytorch_mnist_datasets():
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data',
                                   train=True,
                                   transform=transform,
                                   download=True)
    test_dataset = datasets.MNIST(root='./data',
                                  train=False,
                                  transform=transform,
                                  download=True)
    return train_dataset,test_dataset


def load_mnist(bsize,binarize = True,split_valid = False,dynamic_binarization = False):

    train_dataset,test_dataset = get_pytorch_mnist_datasets()
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=bsize,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=bsize,
                                             shuffle=False)                                               
    # preparing data
    x_train = train_loader.dataset.train_data.float().numpy() / 255.
    x_train = np.reshape( x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2] ) )

    y_train = np.array( train_loader.dataset.train_labels.float().numpy(), dtype=int)

    x_test = test_loader.dataset.test_data.float().numpy() / 255.
    x_test = np.reshape( x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2] ) )

    y_test = np.array( test_loader.dataset.test_labels.float().numpy(), dtype=int)
    
    if binarize:
        x_train = x_train.round()
        x_test = x_test.round()
    # validation set
    
    if split_valid:
        x_val = x_train[50000:60000]
        y_val = np.array(y_train[50000:60000], dtype=int)
        x_train = x_train[0:50000]
        y_train = np.array(y_train[0:50000], dtype=int)
    
    # binarize
    if dynamic_binarization:
        np.random.seed(777)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=bsize, shuffle=True)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=bsize, shuffle=False)
    if split_valid:
        validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
        val_loader = data_utils.DataLoader(validation, batch_size=bsize, shuffle=False)
        data_loaders = train_loader, val_loader, test_loader
    else:
        data_loaders = train_loader, test_loader

    return data_loaders

def load_omniglot(bsize,binarize = True,split_valid = False):
  """Reads in Omniglot images.

  Args:
    binarize: whether to use the fixed binarization

  Returns:
    pytorch data loaders
    train_loader: training images
    val_loader: validation images
    test_loader: test images
  """
  import os
  import urllib
  DATA_DIR = 'data'
  OMNIGLOT = 'omniglot_07-19-2017.mat'
  OMNIGLOT_URL = 'https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat'
  # Get Omniglot
  local_filename = os.path.join(DATA_DIR,OMNIGLOT)
  if not os.path.exists(local_filename):
    os.makedirs(os.path.dirname(local_filename))
    urllib.urlretrieve(OMNIGLOT_URL,local_filename)

  n_validation=1345
  def reshape_data(data):
    return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')

  omni_raw = scipy.io.loadmat('./data/omniglot_07-19-2017.mat')

  x_train = reshape_data(omni_raw['data'].T.astype('float32'))
  x_test = reshape_data(omni_raw['testdata'].T.astype('float32'))
  y_test = np.zeros( (x_test.shape[0], 1) )
  # Binarize the data
  if binarize:
    x_train = x_train.round()
    x_test = x_test.round()


  shuffle_seed = 123
  permutation = np.random.RandomState(seed=shuffle_seed).permutation(x_train.shape[0])
  x_train = x_train[permutation]
  
  if split_valid:
    x_train = x_train[:-n_validation]
    x_val = x_train[-n_validation:]
    
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_val = np.zeros( (x_val.shape[0], 1) )
  
  else:
    x_train = x_train[:-n_validation]
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_test = np.zeros( (x_test.shape[0], 1) )


  # pytorch data loader
  train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
  train_loader = data_utils.DataLoader(train, batch_size=bsize, shuffle=True)
  test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
  test_loader = data_utils.DataLoader(test, batch_size=bsize, shuffle=False)
  if split_valid:
    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=bsize, shuffle=False)
    data_loaders = train_loader, val_loader, test_loader
  else:

    data_loaders = train_loader,test_loader

  return data_loaders
