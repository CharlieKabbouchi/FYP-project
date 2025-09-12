from TTS.api import TTS
import torch
import numpy as np
import sounddevice as sd
import gradio as gr
import os
import librosa
import torchaudio
import soundfile as sf
from scipy.signal import medfilt
from pydub import AudioSegment, effects



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

speakers_dir = "speakers"
os.makedirs(speakers_dir, exist_ok=True)


info1 = sf.info("speakers/en_sample.wav")
print(info1)
# info3 = sf.info("speakers/charlie_test.wav")
# print(info3)
# info4 = sf.info("speakers/charlie_test2.wav")
# print(info4)
# info5 = sf.info("speakers/bis_audio.wav")
# print(info5)


y, sr = librosa.load("speakers/en_sample.wav", sr=None)
print(f"Duration: {len(y)/sr:.2f} sec, Sample Rate: {sr}")
rms = librosa.feature.rms(y=y)[0]
db = librosa.amplitude_to_db(rms, ref=np.max)
print(f"Min dB:   {np.min(db):.2f} dB")
print(f"Max dB:   {np.max(db):.2f} dB")
print(f"Mean dB:  {np.mean(db):.2f} dB")
print(f"Median dB:{np.median(db):.2f} dB")

# info2 = sf.info("speakers/Charlie.wav")
# print(info2)

# y, sr = librosa.load("speakers/Charlie.wav", sr=None)
# print(f"Duration: {len(y)/sr:.2f} sec, Sample Rate: {sr}")
# rms = librosa.feature.rms(y=y)[0]
# db = librosa.amplitude_to_db(rms, ref=np.max)
# print(f"Min dB:   {np.min(db):.2f} dB")
# print(f"Max dB:   {np.max(db):.2f} dB")
# print(f"Mean dB:  {np.mean(db):.2f} dB")
# print(f"Median dB:{np.median(db):.2f} dB")


def clean_backgroundnoise_of_audio(input_path, output_path):
    # this is a simple version to clean background noise (some may still exist)

    y, sr = librosa.load(input_path, sr=None)
    S_full, phase = librosa.magphase(librosa.stft(y))
    noise_power = np.mean(S_full[:, :int(sr * 0.1)], axis=1)
    mask = S_full > noise_power[:, None]
    mask = mask.astype(float)
    mask = medfilt(mask, kernel_size=(1, 5))
    S_clean = S_full * mask
    y_clean = librosa.istft(S_clean * phase)
    sf.write(output_path, y_clean, sr, format='WAV', subtype='FLOAT')


def clean_silence_from_audio(input_path, output_path):
    y, sr = librosa.load(input_path, sr=None)

    # Detect non-silent intervals and put them in a single arr
    intervals = librosa.effects.split(y, top_db=40)
    y_trimmed = np.concatenate([y[start:end] for start, end in intervals])
    sf.write(output_path, y_trimmed, sr, format='WAV', subtype='FLOAT')  # Add these explicit params!


def print_details_of_speaker(speaker_name):
    info2 = sf.info(f"speakers/{speaker_name}.wav")
    print(info2)

    y, sr = librosa.load(f"speakers/{speaker_name}.wav", sr=None)
    print(f"Duration: {len(y)/sr:.2f} sec, Sample Rate: {sr}")
    rms = librosa.feature.rms(y=y)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    print(f"Min dB:   {np.min(db):.2f} dB")
    print(f"Max dB:   {np.max(db):.2f} dB")
    print(f"Mean dB:  {np.mean(db):.2f} dB")
    print(f"Median dB:{np.median(db):.2f} dB")


tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
synth = tts.synthesizer
tts.to(device)


def generate(text, speaker):
    speaker_path = os.path.join(speakers_dir, speaker)
    # generate speech by cloning a voice using default settings
    # tts.tts_to_file(text="It took me quite a long time to develop this.",
    #                 file_path="output.wav",
    #                 speaker_wav="E:/test/example/en_sample.wav",
    #                 language="en")

    # synth = tts.synthesizer

    wav = tts.tts(text=text,
                  speaker_wav=speaker_path,
                  language="en",
                  split_sentences=True
                  )

    # tts.tts_to_file   then play it in gradio
    sd.play(wav, samplerate=24000)
    sd.wait()


def handle_upload(audio_input, text_input_upload):
    if audio_input is None or text_input_upload.strip() == "":
        return None, "Please upload an audio file and enter a speaker name."

    try:
        speaker_name = text_input_upload.strip()
        raw_path = os.path.join(speakers_dir, f"{speaker_name}_raw.wav")
        denoised_path = os.path.join(speakers_dir, f"{speaker_name}_denoised.wav")
        final_path = os.path.join(speakers_dir, f"{speaker_name}.wav")

        waveform, sample_rate = torchaudio.load(audio_input)

        if sample_rate != 24000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)
            waveform = resampler(waveform)

        waveform = waveform.to(torch.float32)
        sf.write(raw_path, waveform.T.numpy().astype(np.float32), samplerate=24000, format='WAV', subtype='FLOAT')

        clean_backgroundnoise_of_audio(raw_path, denoised_path)
        clean_silence_from_audio(denoised_path, final_path)

        os.remove(raw_path)
        os.remove(denoised_path)

        print_details_of_speaker(speaker_name)

        updated_choices = os.listdir(speakers_dir)
        return gr.update(choices=updated_choices, value=os.path.basename(final_path)), f"Saved as {final_path}"

    except Exception as e:
        return None, f"Error processing audio: {str(e)}"


def get_speaker_list():
    return [f for f in os.listdir(speakers_dir)]


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                # this wav format is 16-bit
                file_upload = gr.Audio(label="Upload/Record audio", type="filepath", format="wav") #try type="numpy"
                text_input_upload = gr.Textbox(label="Enter Speaker Name")
                btn_upload = gr.Button("Upload")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)

            with gr.Column():
                text_input = gr.Textbox(label="Enter something")
                speakers = gr.Dropdown(choices=[], value="en_sample.wav", label="Choose a speaker")
                btn = gr.Button("Submit")

        btn_upload.click(fn=handle_upload, inputs=[file_upload, text_input_upload], outputs=[speakers, upload_status])
        btn.click(fn=generate, inputs=[text_input, speakers])

        demo.load(lambda: gr.update(choices=get_speaker_list()), outputs=speakers)
    demo.launch()
