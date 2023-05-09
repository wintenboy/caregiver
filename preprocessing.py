import pandas as pd

def generate_preprocessed(data_path):

    train = pd.read_csv(f'{data_path}/ratings_train.txt', sep='\t')
    test = pd.read_csv(f'{data_path}/ratings_test.txt', sep='\t')

    # 필요없는 열은 drop
    train.drop(['id'], axis=1, inplace=True)
    test.drop(['id'], axis=1, inplace=True)

    # null 제거
    train.dropna(inplace=True)
    test.dropna(inplace=True)

    # 중복 제거
    train.drop_duplicates(subset=['document'], inplace=True, ignore_index=True)
    test.drop_duplicates(subset=['document'], inplace=True, ignore_index=True)

    train.to_csv(f'{data_path}/train_clean.csv', index=False)
    test.to_csv(f'{data_path}/test_clean.csv', index=False)