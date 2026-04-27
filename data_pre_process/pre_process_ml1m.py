#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
import os
import argparse


def pre_process_rating(ml1m_path, processed_path):
    users_inter = {} # {0:[0],1:[]}
    items_inter = {}    # {0:[1],1:[]}
    rating_data = [] # [{'ratingID':1,'asin':0,..},{},{},..]
    rating_data_fl = {} # rating_data split by user {1:[{'ratingID':1,'asin':0,..}]}

    ratings = pd.read_csv(os.path.join(ml1m_path, 'ratings.dat'), sep='::', header=None, engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    rating_data = ratings.values
    grouped = ratings.groupby('UserID')
    rating_data_fl = {user_id -1: group.values -1 for user_id, group in grouped}
    users_inter = {user_id -1: group.values[:,1] -1 for user_id, group in grouped}

    grouped_m = ratings.groupby('MovieID')
    items_inter = {movie_id-1: group.values[:,0]-1 for movie_id, group in grouped_m}

    torch.save(users_inter, os.path.join(processed_path, 'graph_user.pth'))
    torch.save(items_inter, os.path.join(processed_path, 'graph_item.pth'))
    torch.save(rating_data, os.path.join(processed_path, 'ratings.pth'))
    torch.save(rating_data_fl, os.path.join(processed_path, 'ratings_fl.pth'))
    print("ratings processed ->", processed_path)

# movies
def pre_process_movies(ml1m_path, processed_path, model):
    def process_genres(genres):
        split_genres = genres.replace('|', ',')
        return ", belongs to the " + split_genres + " genres."

    def process_title(title):
        split_strings = re.split(r'\s*\(([^)]*)\)\s*$', title)
        title = split_strings[0].strip()
        year = split_strings[1].strip(')') if len(split_strings) > 1 else 'unknown'
        return "A Movie \'" + title + "\', realeased in " + year

    movies = pd.read_csv(os.path.join(ml1m_path, 'movies.dat'), sep='::', encoding='latin1', header=None, engine='python', names=['MovieID', 'Title', 'Genres'])
    movies_data = [""] * 3952
    for i, movie in movies.iterrows():
        movies_data[int(movie['MovieID']) - 1] = process_title(str(movie['Title'])) + process_genres(str(movie['Genres']))

    embedding = model.encode(movies_data, convert_to_tensor=True)
    print(embedding.shape) # 3952*dim
    torch.save(embedding, os.path.join(processed_path, 'items.pth'))
    print("movies processed ->", processed_path)

def pre_process_users(ml1m_path, processed_path, model):
    gender = ['F', 'M']
    gender_feature = ['female', 'male']

    age = [1, 18, 25, 35, 45, 50, 56]
    age_feature = ['under 18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']

    occupation = ["other or not specified", "academic/educator", "artist", "clerical/admin", "college/grad student", "customer service",
                  "doctor/health care", "executive/managerial", "farmer", "homemaker", "K-12 student", "lawyer", "programmer", "retired", 
                  "sales/marketing","scientist", "self-employed", "technician/engineer", "tradesman/craftsman", "unemployed", "writer"]

    users = pd.read_csv(os.path.join(ml1m_path, 'users.dat'), sep='::', header=None, engine='python', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    users_data = [""] * 6040  
    for i, user in users.iterrows():
        users_data[int(user['UserID'])-1] = "A " + gender_feature[gender.index(str(user['Gender']))]+ " user" + ", aged " + age_feature[age.index(int(user['Age']))] + ", with occupation of " + occupation[int(user['Occupation'])] + "."

    embedding = model.encode(users_data, convert_to_tensor=True)
    print(embedding.shape) # 6040*dim
    torch.save(embedding, os.path.join(processed_path, 'users.pth'))
    print("users processed ->", processed_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ml1m-path', type=str, default='./data/ml-1m/', help='Path to raw ML-1m folder (contains users.dat, movies.dat, ratings.dat)')
    parser.add_argument('--processed-path', type=str, default='./data/ml-1m/processed/t5/', help='Output folder for processed .pth files')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-mpnet-base-v2', help='SentenceTransformer model name or local path')
    args = parser.parse_args()

    os.makedirs(args.processed_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device, "model:", args.model)
    model = SentenceTransformer(args.model, device=device)  # loads AFTER args are parsed

    pre_process_users(args.ml1m_path, args.processed_path, model)
    pre_process_movies(args.ml1m_path, args.processed_path, model)
    pre_process_rating(args.ml1m_path, args.processed_path)