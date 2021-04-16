# CS229-Final-Project
Final Project for CS 229: Machine Learning. Project proposal can be found in [proposal/CS_229_Project_Proposal.pdf](/proposal/CS_229_Project_Proposal.pdf).

## Installing

```console
cd python
pip3 install -r requirements.txt
``` 

## Running

To run the training of a sac agent on the CartPole environment for 10000 steps run

```console
python3 python/train_agent.py CartPole-v0 sac --nsteps 10000
``` 