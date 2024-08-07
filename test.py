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

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(768, 64).cuda()
        self.fc2 = nn.Linear(64, 64).cuda()
        self.fc3 = nn.Linear(64, 21).cuda()
        self.activate_func = nn.Tanh().cuda()

    def forward(self, s):
        s = s.cuda()
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        a_prob = torch.softmax(self.fc3(s), dim=1)
        return a_prob

def main(args):
    with open("./"+args.dataset+"/example_test.txt", "rb") as file:
        data = pickle.load(file)
    with open('./'+args.dataset+'/example_test_text.json', 'r') as file:
        txtdata = json.load(file)
    with open('./' + args.dataset + '/hints.json', 'r') as file:
        hint_set = json.load(file)
    tokenizer = BertTokenizer.from_pretrained('JiaqiLee/imdb-finetuned-bert-base-uncased')
    model = BertModel.from_pretrained('JiaqiLee/imdb-finetuned-bert-base-uncased', output_loading_info=False)
    if args.dataset == 'movie':
        movie_df = pd.read_csv('./movie/movies.dat', delimiter='\t', encoding='ISO-8859-1')
        director_df = pd.read_csv('./movie/movie_directors.dat', delimiter='\t', encoding='ISO-8859-1')
        country_df = pd.read_csv('./movie/movie_countries.dat', delimiter='\t', encoding='ISO-8859-1')
        actor_df = pd.read_csv('./movie/movie_actors.dat', delimiter='\t', encoding='ISO-8859-1')
        genre_df = pd.read_csv('./movie/movie_genres.dat', delimiter='\t', encoding='ISO-8859-1')
    elif args.dataset == 'game':
        with open('./game/num_to_title.json', 'r') as file:
            n2t = json.load(file)
        with open('./game/title_to_attr.json', 'r') as file:
            t2a = json.load(file)
    else:
        print('Unknown domain!!')

    def get_embedding_for_games(i):
        def get_game_text(num):
            cats = '; '.join(t2a[n2t[str(num)]]['category'])
            main_cat = t2a[n2t[str(num)]]['main_cat'][0]
            brand = t2a[n2t[str(num)]]['brand'][0]
            felen = len(t2a[n2t[str(num)]]['feature'])
            if felen > 0:
                fe = t2a[n2t[str(num)]]['feature'][random.randint(0, felen - 1)]
            else:
                fe = 'Unknown'
            delen = len(t2a[n2t[str(num)]]['description'])
            if delen > 0:
                de = t2a[n2t[str(num)]]['description'][random.randint(0, delen - 1)]
            else:
                de = 'Unknown'
            text = 'Category: ' + cats + '. Main category: ' + main_cat + '. Brand: ' + brand + '. Feature: ' + fe + '. Description: ' + de + '.'
            return text[:512]

        def generate_txt_embeddings(input_text):
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)  # Averaging token embeddings
        embs = []
        for each in data[i][0]:
            embs.append(generate_txt_embeddings(get_game_text(str(each))))

            return torch.mean(torch.stack(embs), dim=0)  # 768#torch.cat(all_movie_embeddings, dim=0)   # n, 768

        state = get_avg_embedding(data[idx][0])
        return state


    def get_embedding_for_movies(i):  # i is the index of self.data

        def get_genre_sequence(idx, data):
            all_genre = []
            for each in data[idx][0]:
                genre_ = genre_df.loc[genre_df['movieID'] == each, 'genre'].values
                all_genre.append(list(genre_))
            return all_genre

        def get_year_sequence(idx, data):
            all_year = []
            for each in data[idx][0]:
                year_ = movie_df.loc[movie_df['id'] == each, 'year'].values[0]
                all_year.append(year_)
            return all_year

        def get_director_sequence(idx, data):
            director_list = []
            for each in data[idx][0]:
                if each in director_df['movieID'].values:
                    director = director_df.loc[director_df['movieID'] == each, 'directorName'].values[0]
                    director_list.append(director)
            return director_list

        def get_country_sequence(idx, data):
            country_list = []
            for each in data[idx][0]:
                country = country_df.loc[country_df['movieID'] == each, 'country'].values[0]
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
                actors_list = get_actors_by_ranking(each, actor_df)
                actors_lists.append(actors_list)
            return actors_lists

        def get_title_sequence(idx, data):
            all_title = []
            for each in data[idx][0]:
                title_ = movie_df.loc[movie_df['id'] == each, 'title'].values
                all_title.append(list(title_))
            return all_title

        genres_list = get_genre_sequence(i, data)
        production_years_list = get_year_sequence(i, data)
        director_names_list = get_director_sequence(i, data)
        actor_names_list = get_actor_sequence(i, data)
        country_list = get_country_sequence(i, data)
        title_list = get_title_sequence(i, data)

        def prepare_input(genres, director, actors, country, year, title):
            genres_str = ', '.join(genres)
            actors_str = ', '.join(actors)
            input_text = f"Genres: {genres_str}; Director: {director}; Actors: {actors_str}; Country: {country}; Year: {year}; Title: {title}"
            return input_text

        def generate_embeddings(input_text):
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)  # Averaging token embeddings

        all_movie_embeddings = []
        for genres, production_years, director_names, actors, country, title in zip(genres_list, production_years_list,
                                                                             director_names_list, actor_names_list,
                                                                             country_list, title_list):

            input_text = prepare_input(genres, director_names, actors, country, production_years, title)
            embedding = generate_embeddings(input_text)

            all_movie_embeddings.append(embedding)
       # pdb.set_trace()
        return torch.mean(torch.stack(all_movie_embeddings), dim=0)

    openai.api_key = 'YOURKEY'

    model_a = Actor()
    model_a.load_state_dict(torch.load(args.testdir+'.pth'))
    model_a.eval()

    def get_completion(prompt, model="MODEL-NAME"):#can use 'gpt-4'
        messages = [{"role": "assistant", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message["content"]

    for i in range(len(data)):
        prompt = txtdata[i]['prompt']
        if args.dataset == 'movie':
            state = get_embedding_for_movies(i)
        else:
            state = get_embedding_for_games(i)
        logits = model_a(state)
        _, max_index = torch.max(logits, dim=1)
        hint = hint_set[max_index.item()]
        prompt = prompt + ' Hint: ' + hint
        label = txtdata[i]['label']
        for delay_secs in (2 ** x for x in range(0, 10)):
            try:
                response = get_completion(prompt)
                with open(args.output+'.txt', 'a') as file:
                    file.write(f'Label:{label}\nOutput: {response}\n\n\n\n')
                print(i)
                break
            except openai.OpenAIError as e:
                randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                sleep_dur = delay_secs + randomness_collision_avoidance
                print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                time.sleep(sleep_dur)
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for Re2LLM")
    parser.add_argument("--testdir", type=str, default='test_model', help="name-saved-model")
    parser.add_argument("--output", type=str, default='output', help="save LLM outputs")
    parser.add_argument("--dataset", type=str, default='movie', help="movie or game")
    args = parser.parse_args()

    main(args)
