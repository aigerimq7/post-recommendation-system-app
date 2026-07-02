import os

import numpy as np
import pandas as pd

from typing import List
from catboost import CatBoostClassifier
from fastapi import FastAPI
from datetime import datetime

from schema import PostGet
from sqlalchemy import create_engine



def batch_load_sql(query: str):
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )

    conn = engine.connect().execution_options(
        stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=200_000):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)



# функции для выгрузки обученной модели
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("/my/super/path")
    model = CatBoostClassifier(thread_count=8).load_model(model_path)
    return model # здесь не указываем параметры, которые были при обучении, в дампе модели все есть



# функция для выгрузки признаков постов и юзеров
def load_features():
    # Уникальные записи post_id, user_id
    # Где был совершен лайк
    liked_posts_query = """
        SELECT DISTINCT post_id, user_id
        FROM public.feed_data
        WHERE action='like'"""
    liked_posts = batch_load_sql(liked_posts_query).set_index('user_id')

    # Фичи по постам на основе tf-idf
    post_features = pd.read_sql(
        """SELECT * FROM "ajgerim-dubanaeva-mke5439_post_data_fin" """,
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
             "postgres.lab.karpov.courses:6432/startml"  
    )

    # Фичи по юзерам
    user_features = pd.read_sql(
        """SELECT * FROM "ajgerim-dubanaeva-mke5439_user_data_fin" """,
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
             "postgres.lab.karpov.courses:6432/startml"  
    ).set_index('user_id')
    
    # Текст постов
    post_text = pd.read_sql(
        """SELECT post_id, text, topic FROM public.post_text_df """,
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
             "postgres.lab.karpov.courses:6432/startml"  
    )

    return [liked_posts, post_features, user_features, post_text]


model = load_models()
features = load_features()
app = FastAPI()

# основная функция рекомендует посты по id пользователя
def get_recommended_posts(id: int, time: datetime, limit: int) -> List[PostGet]: 
    # Загрузим фичи по пользователям
    user_features = features[2].loc[id]
    user_dict = user_features.to_dict() if isinstance(user_features, pd.Series) else user_features.iloc[0].to_dict()
    
    # Загрузим фичи по постам
    posts_features = features[1]

    # Объединим эти фичи
    for col, val in user_dict.items():
        posts_features[col] = val
        
    # Нужно добавить 'user_id' как признак
    posts_features['user_id'] = id
    
    # Добавим информацию о дате рекомендаций
    posts_features['hour'] = time.hour
    posts_features['month'] = time.month

    # Формируем df для модели
    train_columns = model.feature_names_  
    scoring_df = posts_features[train_columns]

    # Сформируем предсказания вероятности лайкнуть пост для всех постов
    posts_features["predicts"] = model.predict_proba(scoring_df)[:, 1]

    # Отсекаем уже лайкнутые посты
    try:
        liked_posts = features[0].loc[id]
        liked_ids = set(liked_posts['post_id'].tolist() if isinstance(liked_posts, pd.DataFrame) else [liked_posts['post_id']])
    except KeyError:
        liked_ids = set() # Если пользователь еще ничего не лайкал

    filtered_df = posts_features[~posts_features['post_id'].isin(liked_ids)]

    # Выбираем ТОП-посты
    top_posts = filtered_df.nlargest(limit, 'predicts')
    recommended_ids = top_posts['post_id'].tolist()

    # Формируем ответ
    content = features[3][features[3].post_id.isin(recommended_ids)][['post_id', 'text', 'topic']]
    content = content.rename(columns={'post_id': 'id'}).set_index('id').loc[recommended_ids].reset_index()

    return content.to_dict(orient='records')


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, limit=5) -> List[PostGet]:
    limit = int(limit)
    current_time = datetime.now()
    return get_recommended_posts(id, current_time, limit)
