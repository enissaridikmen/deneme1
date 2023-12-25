import pandas as pd
import numpy as np
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
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from googlesearch import search

from pydub import AudioSegment
import numpy as np


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
#pd.set_option('display.max_rows', None)
#pd.set_option('display.width', 500)

df = pd.read_csv("Müzik/music_data_son.csv")




def important_variables(dataframe):
    numeric_columns = dataframe.select_dtypes(include=[float, int]).columns.tolist()
    data = dataframe[numeric_columns]
    range_values = data.max() - data.min()
    sorted_data = range_values.sort_values(ascending=False)
    min_max_diff_df = pd.DataFrame({'Değişken': sorted_data.index, 'Max_Min_Fark': sorted_data.values})
    pd.options.display.float_format = '{:,.5f}'.format
    onemli_degiskenler = min_max_diff_df.iloc[:, 0].tolist()
    return onemli_degiskenler



# def important_variables(dataframe, correlation_threshold = 0.5, drop=False):
#     if drop:
#         correlation_matrix = dataframe.drop([i for i in dataframe.columns if df[i].dtype == 'O'], axis=1).corr()
#     else:
#         correlation_matrix = dataframe.corr()
#     important_variables = set()
#     for column in correlation_matrix.columns:
#         correlated_variables = correlation_matrix.index[correlation_matrix[column] > correlation_threshold].tolist()
#         important_variables.add(column)
#         important_variables.update(correlated_variables)
#     onemli_degiskenler = sorted(important_variables, key=lambda x: sum(correlation_matrix[x] > correlation_threshold),reverse=True)
#     return onemli_degiskenler

#
# onemli_degiskenler = important_variables(df)
#
# girdi = 'enis'

def normalize_et(girdi,onemli_degiskenler):
    girdi = girdi.lower().replace(" ", "").replace("w", "v").replace("q", "k").replace("x", "h")
    mapping = {k: v for v, k in enumerate("abcçdefgğhıijklmnoöprsştuüvyz", 1)}
    girdi_list = []
    sayi_list = []
    for i in girdi:
        girdi_list.append(mapping[i])
    for j in range(len(girdi_list)):
        if j == 13:
            break
        else:
            girdi_list[j] = df[onemli_degiskenler[j]].min() + girdi_list[j] / (29 - 0) * (df[onemli_degiskenler[j]].max() - df[onemli_degiskenler[j]].min())
            sayi_list.append(girdi_list[j])
    return sayi_list

# girdi_normalize = normalize_et(girdi,onemli_degiskenler)


def model_set(dataframe,variable,sayi_list,onemli_degiskenler):
    model = pd.DataFrame(cosine_similarity(dataframe[onemli_degiskenler[:len(sayi_list)]], [sayi_list]).tolist()).sort_values(by=0, ascending=False)
    top_10_df = model.iloc[:10, :].copy()
    top_10_df.columns = ['Cosine Similarity Score']
    top_10_df.reset_index(inplace=True)
    top_10_df.rename(columns={'index': 'Top 10 Index'}, inplace=True)
    pd.set_option('display.float_format', lambda x: f'{x:.10f}')
    top_10_df['Scaled-Cosine Similarity Score'] = (top_10_df['Cosine Similarity Score'] - top_10_df['Cosine Similarity Score'].min()) / (top_10_df['Cosine Similarity Score'].max() - top_10_df['Cosine Similarity Score'].min())
    top_10_index = top_10_df['Top 10 Index'].tolist()
    new_df = pd.DataFrame(pd.DataFrame(dataframe[variable]).loc[top_10_index,variable])
    new_df_no_duplicates = new_df.drop_duplicates(subset=variable)
    return new_df_no_duplicates



# new_df_no_duplicates = model_set(df,"Dosya",girdi_normalize,onemli_degiskenler)
# clean_df yerine akış bozulmasın diye new_df_no_duplicates kullandım
def clean_df(dataframe, new_df_no_duplicates,threshold=0.9):
    clean_df = dataframe.loc[new_df_no_duplicates.index]
    numerical_columns = clean_df.select_dtypes(include=['float64']).columns
    rows_to_drop = []
    for i in range(len(clean_df)):
        for j in range(i + 1, len(clean_df)):
            if all(abs(clean_df.loc[clean_df.index[i], numerical_columns] - clean_df.loc[clean_df.index[j], numerical_columns]) < threshold):
                rows_to_drop.append(clean_df.index[j])
    clean_df = clean_df.drop(index=rows_to_drop).reset_index(drop=True)
    return clean_df

# new_df_no_duplicates=clean_df(df,new_df_no_duplicates)
# clean_df yerine akış bozulmasın diye new_df_no_duplicates kullandım
def file_names(new_df_no_duplicates,variable,threshold=0.7):
    file_names = new_df_no_duplicates[variable].tolist()
    vectorizer = CountVectorizer()
    file_name_matrix = vectorizer.fit_transform(file_names)
    cosine_similarities = cosine_similarity(file_name_matrix, file_name_matrix)
    rows_to_drop = set()
    for i in range(len(cosine_similarities)):
        for j in range(i + 1, len(cosine_similarities)):
            if cosine_similarities[i, j] > threshold:  # Eşik değeri olarak 0.7 kabul ettim.
                rows_to_drop.add(min(i, j))  # İki indeksten daha kısa olanı tut
    new_df_no_duplicates = new_df_no_duplicates.drop(index=rows_to_drop).reset_index(drop=True)
    return new_df_no_duplicates

#
# new_df_no_duplicates = file_names(new_df_no_duplicates,"Dosya")
# play(new_df_no_duplicates,"proje")
# clean_df yerine akış bozulmasın diye new_df_no_duplicates kullandım
def data_prep(dataframe, girdi,variable):
    onemli_degiskenler = important_variables(dataframe)
    sayi_list = normalize_et(girdi,onemli_degiskenler)
    new_df_no_duplicates = model_set(dataframe, variable, sayi_list, onemli_degiskenler)
    new_df_no_duplicates = clean_df(dataframe,new_df_no_duplicates,threshold=0.9)
    new_df_no_duplicates = file_names(new_df_no_duplicates,variable,threshold=0.7)
    return new_df_no_duplicates

# new_df_no_duplicates = data_prep(df, girdi, 'filename')
def create_music_list(new_df_no_duplicates):
    music_list = [j[0] for i, j in enumerate(new_df_no_duplicates.values.tolist()) if i < 3]
    return music_list
    # music_folder = path
    # music_list = [music_folder + "/" + i for i in liste]
    # return music_list
#
# music_list = create_music_list(new_df_no_duplicates)
#
#
# for i in music_list:
#     print(i)


def Waveform(music):
    y, sr = librosa.load(music)

    # Librosa.display.waveshow ile ses dalgasını görselleştir
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    # Matplotlib plotunu Streamlit'e göm
    st.pyplot(fig)

# proje(df,'enis', 'filename',"proje")


degıskenler =  {"Dosya": "Şarkı isimleri",
                "Length": "Şarkının uzunluğu (saniye cinsinden).",
                "Chroma Stft Mean": "Chromagram'ın 12 nota göre ortalama değeri. Chromagram, müzikteki tonları temsil eden bir özelliktir.",
                "Chroma Stft Var": "Chromagram'ın 12 nota göre varyansı. Tonların müzikteki çeşitliliğini ölçer.",
                "RMS Mean": "Root Mean Square (RMS) değerinin ortalama değeri. Ses sinyalinin genel enerjisini temsil eder.",
                "RMS Var": "RMS değerinin varyansı. Ses enerjisinin zaman içindeki değişikliklerini ölçer.",
                "Spectral Centroid Mean": "Spektral merkezin ortalama değeri. Sesin spektral ağırlık merkezini ifade eder.",
                "Spectral Centroid Var": "Spektral merkezin varyansı. Sesin spektral ağırlık merkezindeki değişiklikleri ölçer.",
                "Spectral Bandwidth Mean": "Spektral genişliğin ortalama değeri. Ses spektrumunun yayılma özelliğini temsil eder.",
                "Spectral Bandwidth Var":"Spektral genişliğin varyansı. Spektrumun genişliğindeki değişiklikleri ölçer.",
                "Rolloff Mean":"Spektrumun toplam enerjisinin belirli bir yüzdesini içeren frekansın ortalama değeri.",
                "Rolloff Var":"Rolloff değerinin varyansı. Spektrumun enerjisinin dağılımındaki değişiklikleri ölçer.",
                "Zero Crossing Rate Mean":"Sıfır geçiş oranının ortalama değeri. Ses sinyalinin sıfır çizgisini kaç kez geçtiğini ölçer.",
                "Zero Crossing Rate Var":"Sıfır geçiş oranının varyansı. Ses sinyalinin sıfır çizgisini geçişlerindeki değişiklikleri ölçer.",
                "Tempo":"Şarkının tempoyu temsil eden bir özellik. Tempoyu BPM (vuruçlar per dakika) cinsinden ifade eder.",
                "Music Genre":"Şarkının müzik türünü ifade eden etiket. Bu etiket, veri setinizde ki her şarkının hangi müzik türüne ait olduğunu belirten değişkendir."}




# def play(new_df_no_duplicates,path):
#     liste = [j[0] for i, j in enumerate(new_df_no_duplicates.values.tolist()) if i < 3]
#     music_folder = path
#     music_list = [music_folder + "/" + i for i in liste]
#     mixer.init()
#     mixer.music.set_volume(0.7)
#     for i in range(len(music_list)):
#         print("********************************************************************")
#         madalya = ["🥇", "🥈", "🥉"]
#
#         # Dosya adının yanına tam yolu ekleyin ve sadece dosya adını alın
#         full_path = music_list[i]
#         music_name = re.sub(r'^\s*[0-9]+\s*\.\s*', '',os.path.basename(full_path))  # Başındaki sayıları ve noktaları kaldır
#         music_name = re.sub(r'^\s*\d+\s*', '', music_name)  # Sadece başındaki sayıları kaldır
#         music_name = re.sub(r'^\s*-\s*', '', music_name)  # Başındaki tire işaretini kaldır
#
#         print(f"{i + 1}. Öneri {madalya[i]}: {music_name} dinliyorsunuz 🎶")
#
#         mixer.init()  # mixer'ı başlatın (yeni bir inisalizasyon)
#         mixer.music.load(full_path)
#         mixer.music.play(start=10)
#         time.sleep(15)
#         mixer.music.stop()
#         print("                                                 Müzik bitti 🔇")
#
#     print("********************************************************************")
#     print("Copyright by DATABand \sizin için çalmaya devam edeceğiz... 🎼")
#     print("********************************************************************")


# def equalize_audio(audio_file, gain):
#     # Ses dosyasını yükle
#     sound = AudioSegment.from_file(audio_file)
#
#     # Ses seviyesini eşitle
#     samples = np.array(sound.get_array_of_samples())
#     equalized_samples = samples * gain
#
#     # Yeni bir ses dosyası oluştur
#     equalized_sound = AudioSegment(
#         equalized_samples.tobytes(),
#         frame_rate=sound.frame_rate,
#         sample_width=sound.sample_width,
#         channels=sound.channels
#     )
#
#     return equalized_soun

# def visualize_spectrogram(path):
#     # Ses dosyasını yükle
#     y, sr = librosa.load(path)
#
#     # Spektrogram oluştur
#     D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
#
#     # Spektrogramu görselleştir
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
#     plt.colorbar(format='%+2.0f dB')
#     st.pyplot()
