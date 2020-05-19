import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from optimizers import Lookahead, Lamb # Local file.

ZDIM = 3

'''
Encoder.
'''

class Encoder(nn.Module):

   def __init__(self):
      super().__init__()

      # The 'ref' parameter will allow seamless random
      # generation on CUDA. It indirectly stores the
      # shape of 'z' but is never updated during learning.
      self.ref = nn.Parameter(torch.zeros(ZDIM))

      # Three hidden layers.
      self.hidden_layers = nn.Sequential(
         # First hidden layer.
         nn.Linear(135, 64),
         nn.ReLU(),
         nn.Dropout(p=0.1),
         nn.LayerNorm(64),

         # Second hidden layer.
         nn.Linear(64, 32),
         nn.ReLU(),
         nn.Dropout(p=0.1),
         nn.LayerNorm(32),

         # Third hidden layer.
         nn.Linear(32, 16),
         nn.ReLU(),
         nn.Dropout(p=0.1),
         nn.LayerNorm(16),

         # Welcome to the latent space.
         nn.Linear(16, ZDIM),
      )

   def rnd(self, nsmpl):
      one = torch.ones_like(self.ref)  # On the proper device.
      return Normal(0. * one, one).sample([nsmpl])

   def forward(self, x):
      return self.hidden_layers(x)


'''
Decoder.
'''

class Decoder(nn.Module):

   def __init__(self):
      super().__init__()

      # The 'ref' parameter will allow seamless random
      # generation on CUDA. It indirectly stores the
      # shape of 'z' but is never updated during learning.
      self.ref = nn.Parameter(torch.zeros(ZDIM))

      # Three hidden layers.
      self.hidden_layers = nn.Sequential(
         # First hidden layer.
         nn.Linear(ZDIM, 16),
         nn.ReLU(),
         nn.LayerNorm(16),

         # Second hidden layer.
         nn.Linear(16, 32),
         nn.ReLU(),
         nn.LayerNorm(32),

         # Third hidden layer.
         nn.Linear(32, 64),
         nn.ReLU(),
         nn.LayerNorm(64),
      )

      self.mu = nn.Linear(64, 135)
      self.sd = nn.Linear(64, 135)


   def forward(self, x):
      h = self.hidden_layers(x)
      return self.mu(h), F.softplus(self.sd(h))


'''
Discriminator.
'''

class Discriminator(nn.Module):

   def __init__(self):
      super().__init__()

      self.layers = nn.Sequential(
         # First hidden layer.
         nn.Linear(ZDIM, 64),
         nn.ReLU(),
         nn.BatchNorm1d(64),

         # Second hidden layer.
         nn.Linear(64, 64),
         nn.ReLU(),
         nn.BatchNorm1d(64),

         # Output.
         nn.Linear(64, 1),
         nn.Sigmoid()
      )

   def forward(self, x):
      return self.layers(x)


'''
Data model.
'''

class qPCRData:

   def __init__(self, path, randomize=True, test=True):
      def keep(line):
         items = line.split()
         # Remove negative controls.
         if items[0] == 'A01': return False
         if items[0] == 'B01': return False
         # Remove positive controls.
         if items[0] == 'G12': return False
         if items[0] == 'H12': return False
         return True
      def fmt(line):
         # Raw data (delta Rn).
         raw = [float(x) for x in line.split()[1:]]
         # Take the diff so that numbers are close to 0.
         diffN1 = [raw[0]]  + [raw[i+1]-raw[i] for i in range(0,44)]
         diffN2 = [raw[45]] + [raw[i+1]-raw[i] for i in range(45,89)]
         diffRP = [raw[90]] + [raw[i+1]-raw[i] for i in range(90,134)]
         return diffN1 + diffN2 + diffRP
      with open(path) as f:
         self.data = [fmt(line) for line in f if keep(line)]
      # Create train and test data.
      if test:
         if randomize: np.random.shuffle(self.data)
         sztest = len(self.data) // 10 # 10% for the test.
         self.test = self.data[-sztest:]
         self.data = self.data[:-sztest]

   def batches(self, test=False, randomize=True, btchsz=32):
      data = self.test if test else self.data
      # Produce batches in index format (i.e. not text).
      idx = np.arange(len(data))
      if randomize: np.random.shuffle(idx)
      if btchsz > len(idx): btchsz = len(idx)
      # Define a generator for convenience.
      for ix in np.array_split(idx, len(idx) // btchsz):
         yield torch.tensor([data[i] for i in ix])


if __name__ == "__main__":

   if sys.version_info < (3,0):
      sys.stderr.write("Requires Python 3\n")

   encd = Encoder()
   decd = Decoder()
   disc = Discriminator()

   if len(sys.argv) > 1:
      encd.load_state_dict(torch.load('gaae-encd-1000.tch'))
      decd.load_state_dict(torch.load('gaae-decd-1000.tch'))
      disc.load_state_dict(torch.load('gaae-disc-1000.tch'))

   data = qPCRData('first.txt')

   with torch.cuda.device(3):
      # Do it with CUDA if possible.
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      if device == 'cuda':
         encd.cuda()
         decd.cuda()
         disc.cuda()

      if len(sys.argv) > 1:
         for batch in data.batches():
            batch = batch.to(device)
            with torch.no_grad():
               mu, sd = decd(encd(batch))
               y = Normal(mu, sd).sample()
            np.savetxt(sys.stdout, y.cpu().numpy(), fmt='%.4f')
         sys.exit()

      lr = 0.001 # The celebrated learning rate

      aopt = torch.optim.Adam(encd.parameters(), lr=lr)
      bopt = torch.optim.Adam(decd.parameters(), lr=lr)
      copt = torch.optim.Adam(disc.parameters(), lr=lr)

      asched = torch.optim.lr_scheduler.MultiStepLR(aopt, [800])
      bsched = torch.optim.lr_scheduler.MultiStepLR(bopt, [800])
      csched = torch.optim.lr_scheduler.MultiStepLR(copt, [800])

      # (Binary) cross-entropy loss.
      loss_clsf = nn.BCELoss(reduction='mean')

      lab = lambda a,b,n: torch.tensor([a,b]).repeat_interleave(n)

      for epoch in range(1024):
         n_train_batches = 0
         closs = dloss = floss = 0.
         encd.train()
         decd.train()
         disc.train()
         for batch in data.batches(btchsz=128):
            n_train_batches += 1
            nsmpl = batch.shape[0]
            batch = batch.to(device)

            # Generate fake (Z0) and real (Z1) samples.
            with torch.no_grad():
               # TODO: is there really no way recycle this??
               noise = torch.randn(batch.shape, device=device) * .05
               Z0 = encd(batch + noise)
            Z1 = encd.rnd(nsmpl)
            X = torch.cat([Z0, Z1], 0)

            # PHASE I: update the discriminator.
            Y = lab(.01, .99, nsmpl).to(device)
            disc_loss = loss_clsf(disc(X).squeeze(), Y)

            copt.zero_grad()
            disc_loss.backward()
            copt.step()

            dloss += float(disc_loss)

            # PHASE II: update the encoder-decoder
            noise = torch.randn(batch.shape, device=device) * .05
            Z0 = encd(batch + noise)
            X = torch.cat([Z0, Z1], 0)
            Y = lab(.99, .01, nsmpl).to(device)
            fool_loss = loss_clsf(disc(X).squeeze(), Y)

            mu, sd = decd(Z0)
            cstr_loss = -Normal(mu, sd).log_prob(batch).sum() / batch.numel()

            floss += float(fool_loss)
            closs += float(cstr_loss)

            loss = fool_loss + cstr_loss

            aopt.zero_grad()
            bopt.zero_grad()
            loss.backward()
            aopt.step()
            bopt.step()

         n_test_batches = 0
         tloss = 0.
         encd.eval()
         decd.eval()
         disc.eval()
         for batch in data.batches(btchsz=1024, test=True):
            n_test_batches += 1
            nsmpl = batch.shape[0]
            batch = batch.to(device)

            # PHASE II: update the encoder-decoder
            with torch.no_grad():
               Z0 = encd(batch) # No noise.
               mu, sd = decd(Z0)
            cstr_loss = -Normal(mu, sd).log_prob(batch).sum() / batch.numel()

            tloss += float(cstr_loss)

         asched.step()
         bsched.step()
         csched.step()

         # Display update at the end of every epoch.
         sys.stderr.write('Epoch %d, disc: %f, fool: %f, cstr: %f, test: %f\n' % \
               (epoch+1, dloss / n_train_batches, floss / n_train_batches,
                  closs / n_train_batches, tloss / n_test_batches))

      # Done, save the networks.
      torch.save(encd.state_dict(), 'gaae-encd-%d.tch' % (epoch+1))
      torch.save(decd.state_dict(), 'gaae-decd-%d.tch' % (epoch+1))
      torch.save(disc.state_dict(), 'gaae-disc-%d.tch' % (epoch+1))
