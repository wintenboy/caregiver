import whisper
import os
import csv
import pandas as pd
import re
import torch
torch.cuda.empty_cache()
model = whisper.load_model('medium',device=('cuda:0'))

def gens():
    designated_month = "202209" #202204와 같이 지정해야 함
    f_list = os.listdir(f'careday/careday/{designated_month}')

    # 파일 목록에서 폴더 이름만 추출합니다.
    folder_list = [folder_name for folder_name in f_list if os.path.isdir(f'careday/careday/{designated_month}/{folder_name}')]

    # 폴더 이름을 숫자 형식으로 정렬합니다.
    folder_list = sorted(folder_list)

    days = folder_list
    len_days = len(days)

    for day in days:
        df = pd.DataFrame(columns=["고객번호", "text"])

        file_path = f"careday/careday/{designated_month}/{day}"
        file_list = os.listdir(file_path)
        file_length = len(file_list)
        transcripts = []
        number_list = []

        for i in range(file_length):
            file_name = file_list[i]

            name, ext = os.path.splitext(file_name)
            tokens = name.split("_")

            number_list.append([tokens[1]])
            try :
                transcribed = model.transcribe(f"careday/careday/{designated_month}/{day}/{file_name}",fp16=False)
                transcripts.append(transcribed['text'])
            except:
                transcripts.append(" ")
                continue


        number_df = pd.DataFrame(number_list, columns = ['고객번호'])
        transcripts_df = pd.DataFrame(transcripts, columns=['text'])
        df = pd.concat([number_df, transcripts_df], axis=1)
        globals()[f"df_{day}"] = df
        globals()[f"df_{day}"].to_csv(f"df_{day}.csv", index=False)
        print(f"{day}_transcription done")
    print("DONE!")

gens()


