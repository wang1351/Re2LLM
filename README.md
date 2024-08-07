# Re2LLM
This is PyTorch implementation of our submission [Re2LLM: Reflective Reinforcement Large Language Model for
Session-based Recommendation].

## Architecture of Re2LLM
![image](./main.png)

## Prerequisites
- Python >= 3.6
- PyTorch == 2.1.0
- transformers == 4.30.2
- numpy 1.24.4
- openai == 0.28.0
- gym == 0.21.0

## Data Preparation
- Run data_generation.ipynb 
- Movie dataset available at https://grouplens.org/datasets/hetrec-2011/
- Game dataset available at https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
- Example datasets are provided
- Set your own OpenAI API Key and backbone model before running the following commands.


## Reflective Exploration Module
- Run Reflective Exploration.ipynb

## Reinforcement Utilization Module

### Training

```
python PPO_discrete_main.py --savedir {path_to_test_model} --dataset {movie, game}
```

## Testing

- To test the model and save the LLM output, run the following command:

```
python test.py --testdir {path_to_test_model (e.g., test_model)} --output {path_to_output} --dataset {movie, game}
```
