import os
import pandas as pd

directory = '202206'  # 가져올 csv 파일이 있는 디렉토리 경로
df_list = []  # csv 파일을 저장할 DataFrame 리스트

# 디렉토리 내 모든 파일을 검색하여 csv 파일만 리스트에 추가
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        csv_path = os.path.join(directory, filename)
        print(csv_path)
        df = pd.read_csv(csv_path)
        date_str = os.path.basename(csv_path).split('_')[-1].split('.')[0]  # 파일 이름에서 날짜 부분 추출
        date = pd.to_datetime(date_str + f'-{directory[:4]}', format='%m%d-%Y')  # 날짜 정보를 datetime으로 변환하여 DataFrame에 추가
        df['date'] = date
        df_list.append(df)

# 모든 DataFrame을 하나로 병합
merged_df = pd.concat(df_list, axis=0, ignore_index=True)

# 병합된 DataFrame 사용
print(merged_df.head())