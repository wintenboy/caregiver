import numpy as np
import pandas as pd
import openai
import os
from get_completion import get_completion
import argparse
import re





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default='202204',
                        help='where to data path')
    parser.add_argument('--api_key',
                        type=str,
                        default='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',
                        help='my own api key')
    args = parser.parse_args()

    openai.api_key = args.api_key
    text_datas = pd.read_csv(f'{args.data_path}/df_{args.data_path}.csv')
    phone_nums = list(text_datas['고객번호'].astype('str'))
    texts = text_datas["text"]

    phone_num_ls = []
    proofread_response_ls = []
    keyword_ls = []
    texts_ls = []
    for phone_num, text in list(zip(phone_nums[:3],texts.loc[:3])):
        list_of_slang = {'술기': ['간병인이 가지고 있는 기술을 칭함'],
                        '전원': ['간병인이 다른 병원으로 옮기는다는 뜻'],
                        '케어한 하루': ['간병인 업체 브랜드명'],
                        '서울 성심 간병인회': ['간병인 업체 브랜드명'],
                        '성모 병원': ['자주 언급되는 병원명'],
                        '은평 성모': ['자주 언급되는 병원명'],
                        '여의도 성모 병원':['자주 언급되는 병원명']}

        prompt = f"""
        너가 해야할 것은 아래의 두 사람의 통화내용을 듣고 다음의 두가지 임무를 수행하는 것이야.
        임무 1: 통화 내용 교정하기
        임무 3: 중요한 단어 5개 추출하기

        그리고 다음의 몇가지 주의 사항을 따라.
        주의 사항 :
        통화 내용을 수정하면 안되고, 맞춤법과 은어 위주만 수정해.
        거의 모든 경우에는 말의 첫마디는 서울성심간병인회를 말할 확률이 높으니까 참고하도록 해.
        가끔 첫마디에 케어한 하루를 언급하거나 SKT, KT, LG 와 같이 통신사와 관련된 내용이 나오기도 해.
        통화내용을 교열할 때 대화 내용에서 누가 고객이고 직원인지 구분하지마.
        교정된 대화 내용을 줄바꿈을 사용해서 표현하지마  
        통화내용은 간병인 업체의 직원과 고객과의 대화내용이야.
        대화내용에는 아래와 같은 은어가 사용될 수 있어.
        키워드는 '병원,욕창,술기' 처럼 ,를 사용해서 표현해줘 
        
        은어 목록: ```{list_of_slang}```\

        통화 내용: ```{text}```\
        그리고 대답의 양식은 다음과 같은 방식을 따라.
        -교정된 대화 내용:
        -중요한 단어:
        """
        response = get_completion(prompt)
        pattern = r"(-교정된 대화 내용:|-중요한 단어:)"
        split_response = re.split(pattern, response)
        proofread_response = split_response[2]
        keyword = split_response[4]

        phone_num_ls.append(phone_num)
        texts_ls.append(text)
        proofread_response_ls.append(proofread_response)
        keyword_ls.append(keyword)
    phone_num_df = pd.DataFrame(phone_num_ls,columns=['phone_num'])
    texts_df = pd.DataFrame(texts_ls,columns=['original_text'])
    proofread_df = pd.DataFrame(proofread_response_ls, columns=['proofread'])
    keyword_df = pd.DataFrame(keyword_ls, columns=['keyword'])
    result = pd.concat([phone_num_df,texts_df,proofread_df,keyword_df], axis=1)
    result.to_csv('caregiver_data/examples.csv',index=False)

if __name__ == '__main__':
    main()