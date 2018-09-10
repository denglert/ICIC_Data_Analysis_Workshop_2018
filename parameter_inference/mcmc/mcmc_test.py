#!/usr/bin/env python

import pandas as pd
import numpy as np
import collections
import mcmc.mcmc as mcmc


Theta = collections.namedtuple('theta', ['alpha', 'a', 'b', 'c', 'd', 'e', 'f'])


def physical_model(p, d):


    Dx = p.a*d.x + p.b*d.y + p.c + p.alpha*d.Ex
    Dy = p.d*d.x + p.e*d.y + p.f + p.alpha*d.Ey

    return Dx, Dy


def log_likelihood(p, dataset, vars):

    
    log_likelihood = 0.0

    for idx, d in dataset.iterrows():

        Dx_pred, Dy_pred = physical_model(p, d)

        ll = ((d.Dx_obs - Dx_pred)/vars['sigma_Dx'])**2 + ((d.Dy_obs - Dy_pred)/vars['sigma_Dy'])**2
        log_likelihood += ll

    return log_likelihood


def proposal_function(x):
    
    scale = 0.001

    xnew_l = []
    for i, xi in enumerate(range(len(x))):
        dx = np.random.normal(scale=scale) 
        xnew_i = dx + xi
        xnew_l.append(xnew_i)

    xnew = Theta(*xnew_l)

    return xnew

#######################################################################################

stars = pd.read_csv('eddington.dat', delim_whitespace=True, header=1, skiprows=0)

stars['Dx_obs'] = stars['Dx_obs_uncorrected'] + 1.5
stars['Dy_obs'] = stars['Dy_obs_uncorrected'] + 1.324


parameters = {
                'alpha' : 1.75/19.8, 
                'a'     : 0.0,
                'b'     : 0.0,
                'c'     : 0.0,
                'd'     : 0.0,
                'e'     : 0.0,
                'f'     : 0.0,
              }


vars = {
          'sigma_Dx' : 0.05,
          'sigma_Dy' : 0.05
       }



#pars = mcmc.Parameters(parameters)
theta = Theta(alpha = 1.75/19.8,
              a = 0.0,
              b = 0.0,
              c = 0.0,
              d = 0.0,
              e = 0.0,
              f = 0.0
             )

model = mcmc.Model(log_likelihood, data=stars)
model.set_theta(theta)
model.set_vars(vars)

ll = model.log_likelihood(theta, stars, model.vars)



print('')
print('')
print('sampler initialised.')
print('')
sampler = mcmc.MetropolisHastings(model, proposal_function, theta)

chain_length = 100

print("----------------------------")
print("--- Sampler has started. ---")
print("----------------------------")

for it in range(chain_length):
    print("it: {}".format(it))
    sampler.update()
