import torch
import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # consider gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ConstantKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(self, x, y, n_epochs=100):
        self.train()
        likelihood = self.likelihood
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)
        for i in range(n_epochs):
            optimizer.zero_grad()
            output = self(x)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()
