from pathlib import Path
import queue
import time

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import openai
import pydub
from moviepy import VideoFileClip
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

TEMP_DIR = Path(__file__).parent / 'temp'
TEMP_DIR.mkdir(exist_ok=True)
AUDIO_TEMP_FILE = TEMP_DIR / 'audio.mp3'
VIDEO_TEMP_FILE = TEMP_DIR / 'video.mp4'
MIC_TEMP_FILE = TEMP_DIR / 'mic.mp3'

client = openai.OpenAI()

def transcribe_audio(audio_path, prompt):
    with open(audio_path, 'rb') as audio_file:
        transcription = client.audio.transcriptions.create(
            model='whisper-1',
            language='en',
            response_format='text',
            file=audio_file,
            prompt=prompt,
        )
    return transcription

def add_audio_chunk(audio_frames, audio_chunk):
    for frame in audio_frames:
        sound = pydub.AudioSegment(
            data=frame.to_ndarray().tobytes(),
            sample_width=frame.format.bytes,
            frame_rate=frame.sample_rate,
            channels=len(frame.layout.channels)
        )
        audio_chunk += sound
    return audio_chunk

def save_audio_from_video(video_bytes, audio_output_path):
    with open(VIDEO_TEMP_FILE, mode='wb') as video_file:
        video_file.write(video_bytes.read())
    video_clip = VideoFileClip(str(VIDEO_TEMP_FILE))
    video_clip.audio.write_audiofile(str(audio_output_path))

@st.cache_data
def get_ice_servers():
    return [{'urls': ['stun:stun.l.google.com:19302']}]

def transcribe_mic_tab():
    prompt = st.text_input('(optional) Enter your prompt', key='input_mic')
    webrtc_ctx = webrtc_streamer(
        key='receive_audio',
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={'video': False, 'audio': True}
    )

    if not webrtc_ctx.state.playing:
        st.write(st.session_state.get('mic_transcription', ''))
        return

    container = st.empty()
    container.markdown('Start speaking...')
    audio_chunk = pydub.AudioSegment.empty()
    last_transcription_time = time.time()
    st.session_state['mic_transcription'] = ''
    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                continue
            audio_chunk = add_audio_chunk(audio_frames, audio_chunk)

            now = time.time()
            if len(audio_chunk) > 0 and now - last_transcription_time > 10:
                last_transcription_time = now
                audio_chunk.export(MIC_TEMP_FILE)
                transcription = transcribe_audio(MIC_TEMP_FILE, prompt)
                st.session_state['mic_transcription'] += transcription
                container.write(st.session_state['mic_transcription'])
                audio_chunk = pydub.AudioSegment.empty()
        else:
            break

def transcribe_video_tab():
    prompt = st.text_input('(optional) Enter your prompt', key='input_video')
    video_file = st.file_uploader('Upload a .mp4 video file', type=['mp4'])
    if video_file is not None:
        save_audio_from_video(video_file, AUDIO_TEMP_FILE)
        transcription = transcribe_audio(AUDIO_TEMP_FILE, prompt)
        st.write(transcription)

def transcribe_audio_tab():
    prompt = st.text_input('(optional) Enter your prompt', key='input_audio')
    audio_file = st.file_uploader('Upload a .mp3 audio file', type=['mp3'])
    if audio_file is not None:
        transcription = transcribe_audio(audio_file, prompt)
        st.write(transcription)

def main():
    st.header(' Transcript🎙️', divider=True)
    st.markdown('#### Transcribe audio from microphone, videos, and audio files')
    tab_mic, tab_video, tab_audio = st.tabs(['Microphone', 'Video', 'Audio'])
    with tab_mic:
        transcribe_mic_tab()
    with tab_video:
        transcribe_video_tab()
    with tab_audio:
        transcribe_audio_tab()

if __name__ == '__main__':
    main()