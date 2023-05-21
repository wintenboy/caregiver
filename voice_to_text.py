import whisper
import os
import csv
import pandas as pd
import re
import torch
torch.cuda.empty_cache()
model = whisper.load_model('large',device=('cuda:0'))

def main():
    designated_month = "202209"
    f_list = os.listdir(f'careday/careday/{designated_month}')


    folder_list = [folder_name for folder_name in f_list if os.path.isdir(f'careday/careday/{designated_month}/{folder_name}')]


    folder_list = sorted(folder_list)

    days = folder_list
    len_days = len(days)

    for day in days:
        df = pd.DataFrame(columns=["variables1", "variables2"])

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


        variables1_df = pd.DataFrame(number_list, columns = ['variables1'])
        variables2_df = pd.DataFrame(transcripts, columns=['variables2'])
        df = pd.concat([variables1_df, variables2_df], axis=1)
        globals()[f"df_{day}"] = df
        globals()[f"df_{day}"].to_csv(f"df_{day}.csv", index=False)
        print(f"{day}_transcription done")
    print("DONE!")

if __name__ == '__main__':
    main()


