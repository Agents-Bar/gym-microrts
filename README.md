# Gym-μRTS (pronounced "gym-micro-RTS")

[<img src="https://img.shields.io/badge/discord-gym%20microrts-green?label=Discord&logo=discord&logoColor=ffffff&labelColor=7289DA&color=2c2f33">](https://discord.gg/DdJsrdry6F)
[<img src="https://github.com/vwxyzjn/gym-microrts/workflows/build/badge.svg">](
https://github.com/vwxyzjn/gym-microrts/actions)
[<img src="https://badge.fury.io/py/gym-microrts.svg">](
https://pypi.org/project/gym-microrts/)


This repository is a fork of Costa's [repository](https://github.com/vwxyzjn/gym-microrts) which provides an OpenAPI gym compatible interface over **μRTS** environment authored by [Santiago Ontañón](https://github.com/santiontanon/microrts). 

**Note** that this repository only provides the environment. To see agents in training and action please see the [original repository](https://github.com/vwxyzjn/gym-microrts).

![Visualisation of an actual game](static/fullgame.gif)

## Technical Paper

Before diving into the code, we highly recommend reading the preprint of our paper: [Gym-μRTS: Toward Affordable Deep Reinforcement Learning Research in Real-time Strategy Games](https://arxiv.org/abs/2105.13807)

### Depreciation note

Note that the experiments in the technical paper above are done with [`gym_microrts==0.3.2`](https://github.com/vwxyzjn/gym-microrts/tree/v0.3.2). As we move forward beyond `v0.4.x`, we are planing to deprecate UAS despite its better performance in the paper. This is because UAS has more complex implementation and makes it really difficult to incorporate selfplay or imitation learning in the future.

## Get Started

```bash
# Make sure you have Java 8.0+ installed
$ pip install gym_microrts --upgrade
```

The quickest way to start is to run and modify provided examples in `examples` directory.
For example, to run `hello_world.py` either move to the `examples` directory and run `python hello_world.py`, or from the root of this repository run `python -m examples.hello_world`.


For running a partial observable example, run the `hello_world_po.py` in this repo.


## Environment Specification

Here is a description of Gym-μRTS's observation and action space:

* **Observation Space.** (`Box(0, 1, (h, w, 27), int32)`) Given a map of size `h x w`, the observation is a tensor of shape `(h, w, n_f)`, where `n_f` is a number of feature planes that have binary values. The observation space used in this paper uses 27 feature planes as shown in the following table. A feature plane can be thought of as a concatenation of multiple one-hot encoded features. As an example, if there is a worker with hit points equal to 1, not carrying any resources, owner being Player 1, and currently not executing any actions, then the one-hot encoding features will look like the following:

   `[0,1,0,0,0],  [1,0,0,0,0],  [1,0,0], [0,0,0,0,1,0,0,0],  [1,0,0,0,0,0]`
   

    The 27 values of each feature plane for the position in the map of such worker will thus be:
    
    `[0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0]`

* **Partial Observation Space.** (`Box(0, 1, (h, w, 29), int32)`) Given a map of size `h x w`, the observation is a tensor of shape `(h, w, n_f)`, where `n_f` is a number of feature planes that have binary values. The observation space for partial observability uses 29 feature planes as shown in the following table. A feature plane can be thought of as a concatenation of multiple one-hot encoded features. As an example, if there is a worker with hit points equal to 1, not carrying any resources, owner being Player 1,  currently not executing any actions, and not visible to the opponent, then the one-hot encoding features will look like the following:

   `[0,1,0,0,0],  [1,0,0,0,0],  [1,0,0], [0,0,0,0,1,0,0,0],  [1,0,0,0,0,0], [1,0]`
   

    The 29 values of each feature plane for the position in the map of such worker will thus be:
    
    `[0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0]`

* **Action Space.** (`MultiDiscrete([hw   6   4   4   4   4   7 a_r])`) Given a map of size `h x w` and the maximum attack range `a_r=7`, the action is an 8-dimensional vector of discrete values as specified in the following table. The first component of the action vector represents the unit in the map to issue actions to, the second is the action type, and the rest of components represent the different parameters different action types can take. Depending on which action type is selected, the game engine will use the corresponding parameters to execute the action. As an example, if the RL agent issues a move south action to the worker at $x=3, y=2$ in a 10x10 map, the action will be encoded in the following way:
    
    `[3+2*10,1,2,0,0,0,0,0 ]`

![image](https://user-images.githubusercontent.com/5555347/120344517-a5bf7300-c2c7-11eb-81b6-172813ba8a0b.png)

## Preset Envs:

Gym-μRTS comes with preset environments for common tasks as well as engaging the full game. Feel free to check out the following benchmark:

* [Gym-μRTS V1 Benchmark](https://wandb.ai/vwxyzjn/action-guidance/reports/Gym-microrts-V1-Benchmark--VmlldzozMDQ4MTU)
* [Gym-μRTS V2 Benchmark](https://wandb.ai/vwxyzjn/gym-microrts/reports/Gym-microrts-s-V2-Benchmark--VmlldzoyNTg5NTA)
* [Gym-μRTS V3 Benchmark](https://wandb.ai/vwxyzjn/rts-generalization/reports/Gym-microrts-V3-Environments--VmlldzoyNzQwNzM)


Below are the difference between the versioned environments

|    | use frame skipping | complete invalid action masking            | issuing actions to all units simultaneously | map size |
|----|--------------------|--------------------------------------------|---------------------------------------------|----------|
| v1 | frame skip = 9     | only partial mask on source unit selection | no                                          | 10x10    |
| v2 | no                 | yes                                        | yes                                         | 10x10    |
| v3 | no                 | yes                                        | yes                                         | 16x16    |


## Developer Guide

Highly suggested to use a different environment than the global.
For example, to set up and activate python's official virtual environment execute
```bash
python -m venv .venv
source .venv/bin/activate
```

This creates `.venv` directory and all packages will be under `.venv/`.

### Submodule

For running tests you might need to checkout `microrts`. Since it's included as a submodule you can check it out using
```bash
git submodule update --init --recursive
```

### Java

To run this environment you need to bundle java code into jar so that jpype can import it.
Once you checkout `microrts` you need to execute the `build.sh` script from within the microrts directory.
As mentioned above, the bundle jar needs to be created with Java 8.
Current LTS is 11 and Sept 2021 will release 17, but Java 8 is supported until 2030 (?!?!).

In case you are using Ubuntu 
you can install java using `sudo apt install openjdk-8-jdk`.
This should work even if you have newer Java version but then you need to switch current java version using
```bash
sudo update-alternatives --config java
sudo update-alternatives --config javac
```
or update link to `/usr/bin/javac` from `/usr/lib/jvm

### Other

Required dev environment
```
# install pyenv
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc

# install python 3.9.5
pyenv install 3.9.5
pyenv global 3.9.5

# install pipx
pip install pipx

# install other dev dependencies
pipx install poetry
pipx instal isort
pipx install black
pipx install autoflake
pipx ensurepath
```


```bash
# install gym-microrts
$ git clone --recursive https://github.com/vwxyzjn/gym-microrts.git && \
cd gym-microrts 
pyenv install -s $(sed "s/\/envs.*//" .python-version)
pyenv virtualenv $(sed "s/\/envs\// /" .python-version)
poetry install
# build microrts
cd gym_microrts/microrts && bash build.sh > build.log && cd ..&& cd ..
python hello_world.py
```

## Known issues

[ ] Rendering does not exactly work in macos. See https://github.com/jpype-project/jpype/issues/906


## Papers written using Gym-μRTS
* CoG 2021: [Gym-μRTS: Toward Affordable Deep Reinforcement Learning Research in Real-time Strategy Games](https://arxiv.org/abs/2105.13807)
* AAAI RLG 2021: [Generalization in Deep Reinforcement Learning with Real-time Strategy Games](http://aaai-rlg.mlanctot.info/papers/AAAI21-RLG_paper_33.pdf), 
* AIIDE 2020 Strategy Games Workshop: [Action Guidance: Getting the Best of Training Agents with Sparse Rewards and Shaped Rewards](https://arxiv.org/abs/2010.03956), 
* AIIDE 2019 Strategy Games Workshop: [Comparing Observation and Action Representations for Deep Reinforcement Learning in MicroRTS](https://arxiv.org/abs/1910.12134), 


