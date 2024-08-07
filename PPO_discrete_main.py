import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete
import pdb
import json
import random
from gym import spaces
import pickle
import pandas as pd
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import warnings
import openai
import os
import time
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import logging
logging.set_verbosity_error()


class GPTenv(gym.Env): #bandit
    def __init__(self, args): #item size
        self.dataset = args.dataset

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32)
        self.state = None
        with open("./"+self.dataset+"/example_train.txt", "rb") as file:
            self.data = list(pickle.load(file))
        with open('./'+self.dataset+'/example_train_text.json', 'r') as file:
            self.txtdata = json.load(file)
        with open('./'+self.dataset+'/example_train_base.txt', 'r', encoding='ISO-8859-1') as file:
            self.base = file.read()
        self.count = None
        self.shufflelist = list(range(500))
        with open('./'+self.dataset+'/hints.json','r') as file:
            self.hint_set = json.load(file)
        self.action_space = spaces.Discrete(len(self.hint_set))
        self._max_episode_steps = 500
        self.tokenizer = BertTokenizer.from_pretrained('JiaqiLee/imdb-finetuned-bert-base-uncased')
        self.model = BertModel.from_pretrained('JiaqiLee/imdb-finetuned-bert-base-uncased', output_loading_info=False)
        if self.dataset == 'movie':
            self.movie_df = pd.read_csv('./movie/movies.dat', delimiter='\t', encoding='ISO-8859-1')
            self.director_df = pd.read_csv('./movie/movie_directors.dat', delimiter='\t', encoding='ISO-8859-1')
            self.country_df = pd.read_csv('./movie/movie_countries.dat', delimiter='\t', encoding='ISO-8859-1')
            self.actor_df = pd.read_csv('./movie/movie_actors.dat', delimiter='\t', encoding='ISO-8859-1')
            self.genre_df = pd.read_csv('./movie/movie_genres.dat', delimiter='\t', encoding='ISO-8859-1')
        elif self.dataset =='game':
            with open('./game/num_to_title.json', 'r') as file:
                self.n2t = json.load(file)
            with open('./game/title_to_attr.json', 'r') as file:
                self.t2a = json.load(file)
        else:
            print('Unknown domain!!')

    def get_reward(self, prompt, label):
        openai.api_key = 'YOURKEY'
        def get_completion(prompt, model="gpt-3.5-turbo"):
            messages = [{"role": "assistant", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0
            )
            return response.choices[0].message["content"]

        for delay_secs in (2 ** x for x in range(0, 10)):
            try:
                prompt = prompt
                response = get_completion(prompt)
                break
            except openai.OpenAIError as e:
                randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                sleep_dur = delay_secs + randomness_collision_avoidance
                print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                time.sleep(sleep_dur)
                continue

        output_list = [line.split('. ')[-1].strip() for line in response.strip().split('\n') if line]
        basetxt = self.base.strip().split('\n\n\n\n')[self.shufflelist[self.count]].split('\nOutput: ')[-1]
        base_list = [line.split('. ')[-1].strip() for line in basetxt.strip().split('\n') if line]
        try:
            basendcg = 1/(np.log2((base_list.index(label)+1)+1))
        except ValueError:
            basendcg = 0
        try:
            ndcg = 1/(np.log2((output_list.index(label)+1)+1))
        except ValueError:
            ndcg = 0
        reward = ndcg - basendcg

        return reward


    def step(self, action):
        flag = False
        hint = ""
        if action != 0:
            hint = " Hint: " + self.hint_set[action]
        prompt = self.txtdata[self.shufflelist[self.count]]['prompt'] +" Use numbered bullet points." + hint
        label = self.txtdata[self.shufflelist[self.count]]['label']
        reward = self.get_reward(prompt, label)
        self.count += 1
        if self.count == 500:
            flag = True
        else:
            if self.dataset == 'movie':
                self.state = self.get_embedding_for_movies(self.shufflelist[self.count])
            elif self.dataset == 'game':
                self.state = self.get_embedding_for_games(self.shufflelist[self.count])

        return self.state, reward, flag, None

    def reset(self):
        random.shuffle(self.shufflelist)
        self.count = 0
        if self.dataset == 'movie':
            self.state = self.get_embedding_for_movies(self.shufflelist[self.count])
        elif self.dataset =='game':
            self.state = self.get_embedding_for_games(self.shufflelist[self.count])
        else:
            print('Unknown domain')

        return self.state

    def close(self):
        return None



    def get_embedding_for_games(self, i):
        def get_game_text(num):
            cats = '; '.join(self.t2a[self.n2t[str(num)]]['category'])
            main_cat = self.t2a[self.n2t[str(num)]]['main_cat'][0]
            brand = self.t2a[self.n2t[str(num)]]['brand'][0]
            felen = len(self.t2a[self.n2t[str(num)]]['feature'])
            if felen > 0:
                fe = self.t2a[self.n2t[str(num)]]['feature'][random.randint(0, felen - 1)]
            else:
                fe = 'Unknown'
            delen = len(self.t2a[self.n2t[str(num)]]['description'])
            if delen > 0:
                de = self.t2a[self.n2t[str(num)]]['description'][random.randint(0, delen - 1)]
            else:
                de = 'Unknown'
            text = 'Category: ' + cats + '. Main category: ' + main_cat + '. Brand: ' + brand + '. Feature: ' + fe + '. Description: ' + de + '.'
            return text[:512]

        def generate_txt_embeddings(input_text):
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)  # Averaging token embeddings
        embs = []
        for each in self.data[i][0]:
            embs.append(generate_txt_embeddings(get_game_text(str(each))))

            return torch.mean(torch.stack(embs), dim=0).numpy()  # 768#torch.cat(all_movie_embeddings, dim=0)   # n, 768

        self.state = get_avg_embedding(self.data[idx][0])
        return self.state


    def get_embedding_for_movies(self, i):  # i is the index of self.data

        def get_genre_sequence(idx, data):
            all_genre = []
            for each in data[idx][0]:
                genre_ = self.genre_df.loc[self.genre_df['movieID'] == each, 'genre'].values
                all_genre.append(list(genre_))
            return all_genre

        def get_year_sequence(idx, data):
            all_year = []
            for each in data[idx][0]:
                year_ = self.movie_df.loc[self.movie_df['id'] == each, 'year'].values[0]
                all_year.append(year_)
            return all_year

        def get_director_sequence(idx, data):
            director_list = []
            for each in data[idx][0]:
                if each in self.director_df['movieID'].values:
                    director = self.director_df.loc[self.director_df['movieID'] == each, 'directorName'].values[0]
                    director_list.append(director)
            return director_list

        def get_country_sequence(idx, data):
            country_list = []
            for each in data[idx][0]:
                country = self.country_df.loc[self.country_df['movieID'] == each, 'country'].values[0]
                country_list.append(country)
            return country_list

        def get_actors_by_ranking(movie_id, dataframe):
            movie_actors = dataframe[dataframe['movieID'] == movie_id]
            sorted_actors = movie_actors.sort_values(by='ranking')
            actor_names_sorted = sorted_actors['actorName'].tolist()
            return actor_names_sorted[:4]

        def get_actor_sequence(idx, data):
            actors_lists = []
            for each in data[idx][0]:
                actors_list = get_actors_by_ranking(each, self.actor_df)
                actors_lists.append(actors_list)
            return actors_lists

        def get_title_sequence(idx, data):
            all_title = []
            for each in data[idx][0]:
                title_ = self.movie_df.loc[self.movie_df['id'] == each, 'title'].values
                all_title.append(list(title_))
            return all_title

        genres_list = get_genre_sequence(i, self.data)
        production_years_list = get_year_sequence(i, self.data)
        director_names_list = get_director_sequence(i, self.data)
        actor_names_list = get_actor_sequence(i, self.data)
        country_list = get_country_sequence(i, self.data)
        title_list = get_title_sequence(i, self.data)

        def prepare_input(genres, director, actors, country, year, title):
            genres_str = ', '.join(genres)
            actors_str = ', '.join(actors)
            input_text = f"Genres: {genres_str}; Director: {director}; Actors: {actors_str}; Country: {country}; Year: {year}; Title: {title}"
            return input_text

        def generate_embeddings(input_text):
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)  # Averaging token embeddings

        all_movie_embeddings = []
        for genres, production_years, director_names, actors, country, title in zip(genres_list, production_years_list,
                                                                             director_names_list, actor_names_list,
                                                                             country_list, title_list):

            input_text = prepare_input(genres, director_names, actors, country, production_years, title)
            embedding = generate_embeddings(input_text)

            all_movie_embeddings.append(embedding)
       # pdb.set_trace()
        return torch.mean(torch.stack(all_movie_embeddings), dim=0).numpy()


def evaluate_policy(env, agent):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


def main(args, number, seed):
    env = GPTenv(args)
    env_evaluate = GPTenv(args)


    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n
    args.max_episode_steps = env._max_episode_steps
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0
    evaluate_rewards = []
    total_steps = 0

    replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)
    while total_steps < args.max_train_steps:
        s = env.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)
            s_, r, done, _ = env.step(a)
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0
            if total_steps % args.evaluate_freq == 0:
                print('evaluation...')
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                if evaluate_num % args.save_freq == 0:
                    np.save('./data_train/PPO_discrete_number_{}_seed_{}.npy'.format(number, seed), np.array(evaluate_rewards))
                    agent.save(evaluate_num)
            else:
                print(f"{int(args.evaluate_freq - total_steps % args.evaluate_freq)} more steps before evaluation")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for Re2LLM")
    parser.add_argument("--max_train_steps", type=int, default=int(5e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=1e2, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=32, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.05, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="tanh activation function")
    parser.add_argument("--savedir", type=str, default='test_model', help="name-saved-model")
    parser.add_argument("--dataset", type=str, default='movie', help="movie or game")

    args = parser.parse_args()
    env_index = 1
    main(args,  number=1, seed=0)
