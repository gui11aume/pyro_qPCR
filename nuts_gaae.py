import numpy as np
import pyro
import pyro.distributions as dist
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.constraints as constraints

from pyro.infer import MCMC, NUTS
from torch.distributions.normal import Normal


from gaae import *

# Turn on internal checks for debugging.
# Available only from Pyro 1.3.0.
try:
   pyro.enable_validation(True)
except AttributeError:
   pass


class GeneratorModel:
   def __init__(self, generator):
      self.genr = generator
      # The following is important for JIT compiling. We need
      # to specify that those parameters are now constant.
      self.genr.requires_grad_(False)

   def __call__(self, obs):
      nsmpl = obs.shape[0]
      zero = torch.zeros_like(self.genr.ref)
      one  = torch.ones_like(self.genr.ref)
      # This plate has to be present in the guide as well.
      with pyro.plate('plate_z', nsmpl):
         # Pyro sample "z" (latent variable).
         z = pyro.sample('z', dist.Normal(zero, one).to_event(1))
      # Push forward through the layers.
      mu, sd = self.genr(z)
      # Remove unobserved variables.
      mux = mu[:,90:]
      sdx = sd[:,90:]
      with pyro.plate('plate_x', nsmpl):
         # Pyro sample "x" (observed variables).
         pyro.sample('x', dist.Normal(mux, sdx).to_event(1), obs=obs)


if __name__ == "__main__":

   if sys.version_info < (3,0):
      sys.stderr.write("Requires Python 3\n")

   genr = Decoder()
   genr.load_state_dict(torch.load('gaae-decd-1024.tch'))
   genr.eval()

   data = qPCRData('second.txt', randomize=False, test=False)

   # Do it with CUDA if possible.
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   if device == 'cuda':
      torch.set_default_tensor_type(torch.cuda.FloatTensor)
      genr.cuda()

   model = GeneratorModel(genr)

   nuts_kernel = NUTS(model, adapt_step_size=True, jit_compile=True)
   mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=2000)
   for batch in data.batches(btchsz=8192, randomize=False, test=False):
      obs = batch[:,90:].to(device)
      mcmc.run(obs)
      z = mcmc.get_samples()['z']
      # Propagate forward and sample observable 'x'.
      with torch.no_grad():
         mu,sd = genr(z)
      for i in range(batch.shape[0]):
         x = Normal(mu[:,i,:90], sd[:,i,:90]).sample()
         orig = batch[i,90:].expand([1000, 45])
         out = torch.cat([x,orig], dim=1)
         np.savetxt(sys.stdout, out.cpu().numpy(), fmt='%.4f')
