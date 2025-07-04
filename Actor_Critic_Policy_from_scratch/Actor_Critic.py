
import numpy as np
import torch
import torch.nn as nn
from torch import optim

class A2C(nn.Module):
    '''
    Actor To Critic Implementation with GAE.

    Args : 
        n_feature : number of features in the observation space
        n_actions : number of actions that can be taken in the environment
        device : the hardware that will be used to store the tensors and train the model
        critic_lr : critic's learning rate
        actor_lr : actor's learning rate
        n_envs : since we're using vertorised environments, thus number of environments is required
    '''
    def __init__(
            self,
            n_features :  int,
            n_actions : int,
            device : torch.device,
            critic_lr : float,
            actor_lr : float,
            n_envs : int,
    ) -> None:
        super().__init__()
        self.device=device
        self.n_envs=n_envs

        #Initializing the critic network structure
        critic_layers=[
            nn.Linear(n_features,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,1),
        ]
        #Initializing the actor network structure
        actor_layers=[
            nn.Linear(n_features,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,n_actions),
        ]

        #Defining the actor and critic networks
        self.critic=nn.Sequential(*critic_layers).to(self.device)
        self.actor=nn.Sequential(*actor_layers).to(self.device)
        #Defining the actor and critic optimizers
        self.critic_optim=optim.RMSprop(self.critic.parameters(),lr=critic_lr)
        self.actor_optim=optim.RMSprop(self.actor.parameters(),lr=actor_lr)

    def forward(self, x:np.ndarray) -> tuple[torch.Tensor , torch.Tensor]:
        '''
        Defines the forward pass through the actor critic network.

        Args : 
            x : a vector of states

        Returns : 
            state_values : A tensor returned by the critic network , of size [n_envs,]
            action_logits_vec : A tensor returned by the actor network , of size [n_envs,n_actions]
        '''
        
        x = torch.Tensor(x).to(self.device)
        state_values=self.critic(x)
        action_logits_vec=self.actor(x)
        return (state_values,action_logits_vec)
    
    def select_action(
            self, x:np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Defined to choose the action based on a gaussian distribution.

        Args : 
            x : a vector of states
        
        Returns : 
            actions : A tensor with the actions , of size [n_steps_per_update, n_envs]
            action_log_probs : A tensor with log-probabilities of the actions , of size [n_steps_per_update, n_envs]
            state_values : A tensor with the state values , or size [n_steps_per_update, n_envs]
            entropy : Returns an entropy tensor of the action distribution, size[n_envs]
        '''
        
        state_values,action_logits=self.forward(x)
        action_pd=torch.distributions.Categorical(
            logits=action_logits
        )
        actions=action_pd.sample()
        action_log_probs=action_pd.log_prob(actions)
        entropy = action_pd.entropy()
        return actions, action_log_probs, state_values, entropy
    
    def get_losses(
            self,
            rewards : torch.Tensor,
            action_log_probs : torch.Tensor,
            value_preds : torch.Tensor,
            entropy : torch.Tensor,
            masks : torch.Tensor,  # Fixed: torch.tensor -> torch.Tensor
            gamma : float,
            lam : float,
            ent_coeff : float,
            device : torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Define the actor and critic loss , mitigated from the paper on GAE (https://arxiv.org/abs/1506.02438)

        Args : 
            rewards : A tensor with the rewards for each time step in the episode , with size [n_steps_per_update,n_envs]
            action_log_probs : A tensor of the log-probabilities of the actions taken at each step , of size [n_steps_per_update,n_envs]
            value_preds : A tensor with the state value predictions for each time step in the episode , with size [n_steps_per_update,n_envs]
            masks : A tensor with a mask for each episode , stops the loss from propogating further if the episode ends , size [n_steps_per_update,n_envs]
            gamma : The discount factor for the rewards
            lam : the GAE hyperparameter. (lam=1 corresponts to Monte-Carlo Sampling and 
                                            lam=0 corresponds to normal TD-learning)
            device : The device to run the model on

        Returns : 
            critic_loss : The critic loss of the minibatch
            actor_loss : The actor loss of the minibatch
        ''' 
        
        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device=device)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration
        actor_loss = (
            -(advantages.detach() * action_log_probs).mean() - ent_coeff * entropy.mean()
        )
        return (critic_loss, actor_loss)
    
    def update_parameters(
            self , critic_loss : torch.Tensor , actor_loss : torch.Tensor
    ) -> None:
        '''
        Updating the parameters based on the optimizers declared

        Args : 
            critic_loss : The critic loss
            actor_loss : The actor loss
        '''

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
