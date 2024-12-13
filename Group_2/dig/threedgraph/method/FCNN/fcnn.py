import torch
import torchani
from torch.nn import Linear, ModuleList, CELU
from torch_scatter import scatter_add
from torch_geometric.data import Data
import torch.nn as nn



class FCNN(torch.nn.Module):
    r"""
        A Fully Connected Neural Network implementation for molecular graph processing. 
        Features are generated from `batch_data.z` and `batch_data.pos`, passed through an embedding 
        function to produce 1008-dimensional features, and processed through the FCNN layers to 
        output `v`, which is aggregated to get `u` using scatter add.
        
        Args:
            in_features (int, optional): Input feature size for the FCNN. (default: :obj:`1008`)
            hidden_channels (list, optional): List of hidden channel sizes. (default: :obj:`[256, 192, 160]`)
            out_channels (int, optional): Output embedding size. (default: :obj:`1`)
    """
    def __init__(self, device, in_features=1008, hidden_channels=[256, 192, 160], out_channels=1, guess=False):
        super(FCNN, self).__init__()
        self.device = device
        self.atomidx2symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
        self.in_features = in_features
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.ani_model=torchani.models.ANI2x()
        # freeze the parameters of the ANI model
        for param in self.ani_model.parameters():
            param.requires_grad = False
        
        self.H_network = self.gen_network(
            in_features, hidden_channels, out_channels).to(device)
        self.C_network = self.gen_network(
            in_features, hidden_channels, out_channels).to(device)
        self.N_network = self.gen_network(
            in_features, hidden_channels, out_channels).to(device)
        self.O_network = self.gen_network(
            in_features, hidden_channels, out_channels).to(device)
        self.F_network = self.gen_network(
            in_features, hidden_channels, out_channels).to(device)
        
        self.H = nn.Parameter(torch.tensor(-0.5), requires_grad=True).to(device)
        self.C = nn.Parameter(torch.tensor(-3.0), requires_grad=True).to(device)
        self.N = nn.Parameter(torch.tensor(-3.5), requires_grad=True).to(device)
        self.O = nn.Parameter(torch.tensor(-4.0), requires_grad=True).to(device)
        self.F = nn.Parameter(torch.tensor(-4.5), requires_grad=True).to(device)
        self.guess = guess
        

    def gen_network(self, in_features, hidden_channels, out_channels):
        """
        Generate the FCNN layers based on the input parameters.
        """
        layers = []
        input_dim = in_features
        for hidden_dim in hidden_channels:
            layers.append(Linear(input_dim, hidden_dim))
            layers.append(CELU(alpha=0.1))
            input_dim = hidden_dim
        layers.append(Linear(input_dim, out_channels))
        return torch.nn.Sequential(*layers)

    def process_dataone(self, data):
        atomidx2symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
        device = data.pos.device
        ani_model = self.ani_model.to(device)
        elements = [str(atomidx2symbol[int(i)]) for i in data.z]
        species = ani_model.species_to_tensor("".join(elements)).unsqueeze(0).to(device) 
        
        coordinates = data.pos.reshape(1, -1, 3)
        species_coordinates = (species, coordinates)
        species_aevs = ani_model.aev_computer(species_coordinates)
        # species_arr = data.z.cpu().numpy()
        aevs_arr = species_aevs
        return aevs_arr

    def embedding(self, z, pos, batch):
        """
        Define a custom embedding function. For example, using a combination of
        z (atomic number embedding) and pos (positional features).
        """
        aevs_tensor_list = []
        for i in range(batch.min().item(), batch.max().item() + 1):
            mask = batch == i
            data = Data(z=z[mask], pos=pos[mask])
            aevs_tensor_ = self.process_dataone(data).aevs.squeeze(0)
            aevs_tensor_list.append(aevs_tensor_)
            
        aevs_tensor = torch.cat(aevs_tensor_list, dim=0)
        return aevs_tensor  

    def forward(self, batch_data):
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        idx = list(range(len(batch)))
        idx = torch.tensor(idx).to(z.device)
        # Generate features from embedding
        if self.guess:
            features = torch.zeros([z.size(0), self.in_features]).to(self.device)
        else:
            features = self.embedding(z, pos, batch) 
        
        v_cat = torch.zeros([features.size(0), self.out_channels]).to(self.device)
        
        models_list = [self.H_network, self.C_network, self.N_network, self.O_network, self.F_network]
        guess_list = [self.H, self.C, self.N, self.O, self.F]
        for i,element in enumerate([1, 6, 7, 8, 9]):
            mask = z == element
            idx_ = idx[mask]
            features_ = features[idx_]
            Model = models_list[i]
            if not self.guess:
                v_ = Model(features_)
            else:
                shape_v = (features_.shape[0], self.out_channels)
                v_ = torch.zeros(shape_v).to(self.device)
                v_ = v_ + guess_list[i]
            v_cat[idx_] = v_
        
        # Pass through the FCNN
        v = v_cat
        
        # Aggregate `v` to compute `u` using scatter add
        u = scatter_add(v, batch, dim=0)

        return u
