import warnings
import pandas as pd
import numpy as np
import streamlit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import pygame
from pygame import mixer
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
from streamlit_chatbox import *
from Proje_3 import *
from googlesearch import search
from pydub import AudioSegment

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
#pd.set_option('display.max_rows', None)
#pd.set_option('display.width', 500)
st.set_page_config(
    page_title="Streamlit Uygulaması",
    page_icon=":smiley:",
    layout="wide",  # wide kullanarak geniş bir düzen seçebilirsiniz
)


@st.cache_data
def get_data():
    df = pd.read_csv("Müzik/music_data_son.csv")
    return df

df = get_data()

warnings.simplefilter(action='ignore', category=Warning)


st.header(":red[DataBand İftiharla Sunar😊]")
tab_giriş, tab_sum, tab_app = st.tabs(["Introduction", "Summary", "App"])


video_file = open("WhatsApp Video 2023-12-25 at 13.27.30_15fbedd8.mp4", 'rb')
video_bytes = video_file.read()
tab_giriş.video(video_bytes, format="video/mp4", start_time = 0)



sol, sag = tab_sum.columns([1,2])

col1,col2,col3 = sag.columns([5, 5, 5])
col1.metric("Gözlem Birimi:", df.shape[0])
col2.metric("Değişken Adet:", len(df.columns))
col3.metric("Şarkı Adet:", df['Dosya'].nunique())

sag.dataframe(df, width=2000)


feature = sol.radio("Deişken Tanımları",df.columns, index=None)

for col in df.columns:
    if feature == col:
        sol.markdown("<div style='width:400px;'><span style='color: red; font-weight: bold;'>{}</div>".format(col), unsafe_allow_html=True)
        sol.markdown("<div style='width:400px;'><span style='color: black; font-weight: bold;'>{}</div>".format(degıskenler[col]), unsafe_allow_html=True)






# sag.markdown("<div style='width:800px;'><span style='color: red; font-weight: bold;'>Length:</span> Şarkının uzunluğu (saniye cinsinden)</div>", unsafe_allow_html=True)
# sag.markdown("<div style='width:800px'><span style='color: red; font-weight: bold;'>Chroma Stft Mean:</span> Chromagram'ın 12 nota göre ortalama değeri. Chromagram, müzikteki tonları temsil eden bir özelliktir.</div>", unsafe_allow_html=True)
# sag.markdown("<div style='width:800px'>Chroma Stft Var: Chromagram'ın 12 nota göre varyansı. Tonların müzikteki çeşitliliğini ölçer.</div>", unsafe_allow_html=True)
# sag.markdown("<div style='width:800px'>RMS Mean: Root Mean Square (RMS) değerinin ortalama değeri. Ses sinyalinin genel enerjisini temsil eder.</div>", unsafe_allow_html=True)
# sag.markdown("<div style='width:800px'>RMS Var: RMS değerinin varyansı. Ses enerjisinin zaman içindeki değişikliklerini ölçer.</div>", unsafe_allow_html=True)
#
# sag.markdown("<div style='width:800px'>Chroma Stft Mean:Chromagram'ın 12 nota göre ortalama değeri. Chromagram, müzikteki tonları temsil eden bir özelliktir.</div>", unsafe_allow_html=True)
# sag.markdown("<div style='width:800px'>Chroma Stft Mean:Chromagram'ın 12 nota göre ortalama değeri. Chromagram, müzikteki tonları temsil eden bir özelliktir.</div>", unsafe_allow_html=True)
# sag.markdown("<div style='width:800px'>Chroma Stft Mean:Chromagram'ın 12 nota göre ortalama değeri. Chromagram, müzikteki tonları temsil eden bir özelliktir.</div>", unsafe_allow_html=True)
# sag.markdown("<div style='width:800px'>Chroma Stft Mean:Chromagram'ın 12 nota göre ortalama değeri. Chromagram, müzikteki tonları temsil eden bir özelliktir.</div>", unsafe_allow_html=True)
# sag.markdown("<div style='width:800px'>Chroma Stft Mean:Chromagram'ın 12 nota göre ortalama değeri. Chromagram, müzikteki tonları temsil eden bir özelliktir.</div>", unsafe_allow_html=True)
# sag.markdown("<div style='width:800px'>Chroma Stft Mean:Chromagram'ın 12 nota göre ortalama değeri. Chromagram, müzikteki tonları temsil eden bir özelliktir.</div>", unsafe_allow_html=True)
# sag.markdown("<div style='width:800px'>Chroma Stft Mean:Chromagram'ın 12 nota göre ortalama değeri. Chromagram, müzikteki tonları temsil eden bir özelliktir.</div>", unsafe_allow_html=True)
# sag.markdown("<div style='width:800px'>Chroma Stft Mean:Chromagram'ın 12 nota göre ortalama değeri. Chromagram, müzikteki tonları temsil eden bir özelliktir.</div>", unsafe_allow_html=True)



# tab_app sekmesi

# Giriş alanını oluşturun
girdi = tab_app.text_input("İsim giriniz")


if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

tab_app.button('Müziğe Kulak Ver!', on_click=click_button)

if st.session_state.clicked:
    tab_app.write(f"{girdi} ismini en iyi anlatan 3 müziği duymaya hazır mısın?")
    new_df_no_duplicates = data_prep(df,girdi,'Dosya')
    music_list = create_music_list(new_df_no_duplicates)
    for i, (music, emoji) in enumerate(zip(music_list, ["🥇", "🥈", "🥉"])):
        music2 = re.sub(r'^\s*[0-9]+\s*\.\s*', '',os.path.basename(music))  # Başındaki sayıları ve noktaları kaldır
        music2 = re.sub(r'^\s*\d+\s*', '', music2)  # Sadece başındaki sayıları kaldır
        music2 = re.sub(r'^\s*-\s*', '', music2)  # Başındaki tire işaretini kaldır
        music2 = re.split(r'\.mp3', music2)[0]
        music2 = music2.replace("_"," ").replace("uÌˆ", "ü").replace("UÌˆ", "Ü").replace("IÌ‡", "İ").replace("Ä±", "ı").replace("sÌ§", "ş").replace("SÌ§", "Ş").replace("gÌ†", "ğ").replace("GÌ†", "Ğ").replace("cÌ§", "ç").replace("CÌ§", "Ç").replace("OÌˆ", "Ö").replace("oÌˆ", "ö")
        if tab_app.button(f"{i + 1}. Öneri {emoji}: {music2} dinliyorsunuz 🎶"):
            # mixer.init()  # mixer'ı başlatın (yeni bir inisalizasyon)
            # mixer.music.load(music)
            # tab_app.write("Waveform gösteriliyor:")
            # with tab_app.container():
            #     Waveform(music)
            # mixer.music.play(start=10)
            if music2:
                google_arama = f"{music2} video klip"
                with tab_app.container():
                    st.write("Google'da Arama Sorgusu:", google_arama)
                for result in search(google_arama,num=1, stop=1, pause=2):
                    if result:
                        tab_app.video(result, start_time=0, format="video/mp4")
            # time.sleep(15)
            # mixer.music.stop()



# streamlit run .\Proje_Çalışması.py