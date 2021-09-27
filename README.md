# MeanField-MDP
This is an implementation of the work **Entropy-Regularized Finite Mean-Field Games** submitted to ACC-22. 
The code structure is as follows: 
The resource allocation example is in the ``ERMFG_paper_example`` folder.
All the environments, including the finite population MDP generation are included in the ``env`` folder. 
The MDP solvers/evaluators and MFG solvers (with and without entropy-regularization) are all in the ``solvers`` folder.

## Notes on the ``MeanFieldEnv``
The reference policy and inverse temperature are defined within the MFG environment. 
To change, play with the parameter sections at the top of the environment code, or use ``set_beta`` and ``set_prior`` functions of the ``MeanFieldEnv``.

If using entropy-regularization, do not forget to set ``MeanFieldEnv.entropy_regularized`` to ``True``.

If using only the terminal rewards, set ``MeanFieldEnv.terminal_reward_only`` to ``True``. This will speed up error verification significantly.


## Data Structure
Data            |Expression         |Type           |Notes
---             |---                |---            |---
n_states        |`env.n_states`     |Int            |
n_actions       |`env.n_actions`    |List           |`N` elements. `env.n_actions[s]` gives the number of actions available at state `s`. 
Transition      |`env.T`            |List           |`env.T[a][s]` gives a `N` array of probability transitioning from `s` to other states. If state `s` does not have action `a`, then an empty list `[]` is returned.
state MF at t   |`mu_t`             |np array       |Length `N`. `mu[s]` gives the measure on `s`
state action MF at t|`nu_t`         |np array       |Length `sum(n_actions)`
state MF flow   |`mu`               |List of np array|Length `Tf+1`. Consists of `mu_t` at each time steps from 0 to `Tf` included.
sa MF flow      |`nu`               |List of np array|Length `Tf+1`. Consists of `nu_t`.

## Exemplary Outputs
The following figure presents the evolution of the population in the resource allocation example.
![Blotto Game Mean Field Flow](/figures/blotto-game-flow.png)

The following figure is a sample output of the error verification algorithm
![Blotto Game Mean Field Flow](/figures/error-verification.png)