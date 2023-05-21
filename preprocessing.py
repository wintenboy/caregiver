import pandas as pd
from sklearn.model_selection import train_test_split
import random
def generate_preprocessed(data_path):
    # 데이터 불러오기
    all_data = pd.read_csv(f'{data_path}/proofread.csv')

    # 중복 및 결측치 제거
    all_data.dropna(inplace=True)
    all_data.drop_duplicates(subset=['variables3'], inplace=True, ignore_index=True)

    # 데이터 분리
    X = all_data[['variables3']]
    Y = all_data[['label']]
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=42)

    train_data = pd.concat([train_x, train_y], axis=1)
    test_data = pd.concat([test_x, test_y], axis=1)

    train_data.to_csv(f'{data_path}/caregiver_train.csv', index=False)
    test_data.to_csv(f'{data_path}/caregiver_test.csv', index=False)


