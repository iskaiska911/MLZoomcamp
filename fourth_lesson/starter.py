#!/usr/bin/env python
# coding: utf-8



import pickle
import pandas as pd
import sys



with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)





def read_data(filename):
    categorical = ['PULocationID', 'DOLocationID']

    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df



def predict(year,month):

    categorical = ['PULocationID', 'DOLocationID']


    print(f"""{month}""")
    print(f"""{year}""")


    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f""" std dev {y_pred.std()}""")


    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df['pred']=y_pred


    df_result=df[['ride_id','pred']]


    df_result.to_parquet(
    'predictions5',
    engine='pyarrow',
    compression=None,
    index=False
)

    print(f"""mean duration {y_pred.mean()}""")

if __name__ == "__main__":
    #year=int(sys.argv[1])
    year=int(2023)
    #month=int(sys.argv[2])
    month=int(5)
    predict(year,month)




