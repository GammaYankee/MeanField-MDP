# MeanField-MDP

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
