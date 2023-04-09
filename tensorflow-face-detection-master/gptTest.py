#!/usr/bin/env python
# coding: utf-8

# ##### STT API를 이용하여 조합한 기능입니다.
# 1. keyBert 키워드 생성
# 2. koGPT2 질문 생성
# 3. 발화 속도 측정 



#KeyBert 관련 import

import warnings
warnings.filterwarnings('ignore')

from multiprocessing import Process, Value
import multiprocessing as mp

import os
# [START speech_transcribe_streaming_mic]
#koGPT2 관련 import
import torch
import transformers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
import fastai

import matplotlib.pyplot as plt

#download KoGPT2 model and tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>') 


model = AutoModelForCausalLM.from_pretrained("/Users/yujeong/Downloads/KoGpt2")   #train한 모델 불러오기


import numpy as np
import itertools

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


speed_list = [0,0,0,0,0]  # 발화 속도 측정용
img_path = os.getcwd()  # 현재 경로에 저장하기 위함


def kogpt2(stt_txt):
    text=stt_txt  # 받아온 발화 텍스트

    inpt_len=len(text)  # 길이 

    input_ids = tokenizer.encode(text)
    gen_ids = model.generate(torch.tensor([input_ids]),
                               max_length=inpt_len + 15,   #15자 생성
                               repetition_penalty=2.0,
                               pad_token_id=tokenizer.pad_token_id,
                               eos_token_id=tokenizer.eos_token_id,
                               bos_token_id=tokenizer.bos_token_id,
                               use_cache=True
                            )
    generated = tokenizer.decode(gen_ids[0,:].tolist())

    #생성된 문장에서 입력한 부분 제외하고
    out=generated[inpt_len:]

    #생성된 질문 출력(첫번째 질문만)
    question=out.split('?')    
    
    f3 = open('/Users/yujeong/Desktop/question.txt', 'w')

    # write 함수를 이용해서 파일에 텍스트 쓰기

    f3.write(question[0]+'?')
#     print(question[0]+'?')  # 찍어보기용

    # 파일 닫기
    f3.close()


def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):

    # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

    # 각 키워드들 간의 유사도
    word_similarity = cosine_similarity(candidate_embeddings)

    # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
    # 만약, 2번 문서가 가장 유사도가 높았다면
    keywords_idx = [np.argmax(word_doc_similarity)]

    # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
    # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # MMR을 계산
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # keywords & candidates를 업데이트
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]


def keyword_ext(doc):

    #형태소 분석기를 통해 명사만 추출한 문서를 만듭니다.
    okt = Okt()

    tokenized_doc = okt.pos(doc)
    tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

    #여기서는 사이킷런의 CountVectorizer를 사용하여 단어를 추출합니다. CountVectorizer를 사용하는 이유는 n_gram_range의 인자를 사용하면 쉽게 n-gram을 추출할 수 있기 때문입니다. 예를 들어, (2, 3)으로 설정하면 결과 후보는 2개의 단어를 한 묶음으로 간주하는 bigram과 3개의 단어를 한 묶음으로 간주하는 trigram을 추출합니다.
    n_gram_range = (2, 3)

    count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
    candidates = count.get_feature_names()

    #한국어를 포함하고 있는 다국어 SBERT 로드
    model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    doc_embedding = model.encode([doc])
    candidate_embeddings = model.encode(candidates)

    keywords = mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.7)
    return keywords  #화면에 출력되도록 함


import re
import sys

from google.cloud import speech

import pyaudio
from six.moves import queue

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

#면접 응답 기록할 stt.txt 파일 열기
f=open("stt.txt",'w')




class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


            
import cv2
def listen_print_loop(responses, pause):
    num_chars_printed = 0
    
    temp = ""
    temp_cnt = 0


    print("발화 시작")  # 찍어보는 용

    for response in responses:

        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:

            alternative = result.alternatives[0]
            print("Transcript: {}".format(alternative.transcript))

            #텍스트 입력 - transcript 결과 text 파일에 저장
            f.write(format(alternative.transcript)+"\n")
            stt_inpt=format(alternative.transcript)

            keylist = keyword_ext(stt_inpt)
        
            kogpt2(stt_inpt)

            f2 = open('/Users/yujeong/Desktop/keywords.txt', 'w')

            # write 함수를 이용해서 파일에 텍스트 쓰기
            
            for line in keylist:
                f2.write(line)

            # 파일 닫기
            f2.close()

            w_cnt=0  #단어수 
            for word_info in alternative.words:
                word = word_info.word

                ###더듬은 횟수만 처리되도록 코드 변경###
                if (temp == word) : 
                    print("더듬음")
                    temp_cnt = 1 + temp_cnt
                if temp_cnt >=2 and word[0] == tmep:
                    print("더듬음")

                temp_cnt = 0
                start_time = word_info.start_time
                end_time = word_info.end_time

                temp = word

                ####한 질문에 대한 응답 끝나면(사용자가 종료 버튼을 누르거나 time out될 경우) 발화 시간 화면에 출력###

                if w_cnt==0:    #말하기 시작 시점
                    strt=start_time.total_seconds()
                total_end=end_time.total_seconds()  #말하기 끝난 시점

                total_time = total_end-strt

                w_cnt+=1
            """ 발화속도 측정  - 한 줄씩 - 

                가) 글자 수로 빠르기 측정 (발화 속도 측정 사이트 & 연구 자료 참조)
                    1. 매우 느림 : ~ 390
                    2. 느림: 390 ~ 420
                    3. 보통: 420 ~ 450
                    4. 빠름: 450 ~ 480
                    5. 매우 빠름: 480 ~ 

            """
            speed = (len(transcript)/total_time) * 60  # 분 당 말한 문자 수로 speed 체크하기 위함 

            
            # 속도 측정 
            if speed < 390: speed_list[0] = speed_list[0] + 1
            elif speed >=390 and speed < 420 : speed_list[1] = speed_list[1] + 1
            elif speed >=420 and speed < 450 : speed_list[2] = speed_list[2] + 1
            elif speed >=450 and speed < 480 : speed_list[3] = speed_list[3] + 1
            elif speed > 480 : speed_list[4] = speed_list[4] + 1


            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            # 발표 종료 버튼 GUI에서 눌렀을 때 멈춘다
            if pause.value == 1:
                print("Exiting..")
                break

            num_chars_printed = 0
            
            



def main2(pause):  # 공유 변수 받아옴
    

    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = "ko-KR"  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
        enable_word_time_offsets=True,  ##dalkkommi
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )
    

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )
        
        responses = client.streaming_recognize(streaming_config, requests)
        # Now, put the transcription responses to use.
        
        listen_print_loop(responses, pause)
        
    
    # 발화속도 측정용 그래프 
    x = np.arange(5)
    speed = ['very slow', 'slow', 'normal', 'fast', 'very fast']  # 발화 속도

    plt.bar(x, speed_list)
    plt.xticks(x, speed)

    plt.savefig(img_path + '/save_fig/save_speed/speed.png')
    plt.clf()







