from moviepy.editor import *
import whisper
from transformers import pipeline

import glob
import torch
import argparse
import logging
import os

from split_paragraph import gen_parag

def extract_transcript(file_path, pipe, save_dir, chunk_length_s=20, sentence_embedding_model="all-minilm-l6-v2", p_size=10, order=10, target_language='en'):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)
    
    file_name = file_path.split('/')[-1].replace('.mp3', '')
    # decode the audio

    generate_kwargs = {"task":"transcribe", "language":f"<|{target_language}|>"}
    prediction = pipe(
        file_path,
        return_timestamps=True,
        chunk_length_s=chunk_length_s,
        stride_length_s=(5),
        generate_kwargs=generate_kwargs
    )

    para_chunks, para_timestamp = gen_parag(
        prediction['chunks'],
        model_name=sentence_embedding_model,
        p_size=p_size,
        order=order
    )
    
    for chunk, timestamp in zip(para_chunks, para_timestamp):
        trans_path = f"{save_dir}/{file_name}_{timestamp[0]}_{timestamp[1]}.txt"
        with open(trans_path, 'w', encoding='utf-8') as f:
            f.write(chunk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper-model", type=str, default="whisper-large-v2")
    parser.add_argument("--clip-duration", type=int, default=120)
    parser.add_argument("--target-language", type=str, default="en")
    parser.add_argument("--sentence-embedding-model", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--chunk-length", type=int, default=20)
    parser.add_argument("--p-size", type=int, default=10)
    parser.add_argument("--order", type=int, default=5)
    args, _ = parser.parse_known_args()
    
    input_dir = "/opt/ml/processing/input"
    transcript_dir = "/opt/ml/processing/transcripts"
    
    clip_duration = args.clip_duration #second
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "automatic-speech-recognition",
        model=f"openai/{args.whisper_model}",
        device=device
    )
    
    mp4_list = glob.glob(input_dir + "/*.mp4")
    mp3_list = glob.glob(input_dir + "/*.mp3")
    file_list = mp4_list + mp3_list
    for file_path in file_list:
        print(file_path)
        if file_path.endswith('.mp4'):
            file_name = file_path.split('/')[-1].replace('.mp4', '')
            
            video = VideoFileClip(file_path)
            file_path = file_path.replace('.mp4', '.mp3')
            video.audio.write_audiofile(file_path)
        elif file_path.endswith('.mp3'):
            file_name = file_path.split('/')[-1].replace('.mp3', '')
        
        trans_dir = f"{transcript_dir}/{file_name}"
        if not os.path.exists(trans_dir):
            os.makedirs(trans_dir)
        
        extract_transcript(
            file_path,
            pipe,
            trans_dir,
            chunk_length_s=args.chunk_length,
            p_size=args.p_size,
            order=args.order,
            sentence_embedding_model=args.sentence_embedding_model,
            target_language=args.target_language
        )
