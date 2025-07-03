from TTS.api import TTS
import torch
import sounddevice as sd
import gradio as gr
import os
import shutil
from pydub import AudioSegment
import librosa
import matplotlib.pyplot as plt
import torchaudio
import tempfile
import soundfile as sf


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

speakers_dir = "speakers"
os.makedirs(speakers_dir, exist_ok=True)


info1 = sf.info("speakers/en_sample.wav")
print(info1)
# info2 = sf.info("speakers/charlie.wav")
# print(info2)
# info3 = sf.info("speakers/charlie_test.wav")
# print(info3)
# info4 = sf.info("speakers/charlie_test2.wav")
# print(info4)
# info5 = sf.info("speakers/bis_audio.wav")
# print(info5)


y, sr = librosa.load("speakers/en_sample.wav", sr=None)
print(f"Duration: {len(y)/sr:.2f} sec, Sample Rate: {sr}")

plt.plot(y)
plt.title("Waveform")
plt.show()

# y, sr = librosa.load("speakers/charlie_test.opus", sr=None)
# print(f"Duration: {len(y)/sr:.2f} sec, Sample Rate: {sr}")

# plt.plot(y)
# plt.title("Waveform")
# plt.show()

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
                    language="en")

    # tts.tts_to_file   then play it in gradio
    sd.play(wav, samplerate=24000)
    sd.wait()


# def handle_upload(audio_input, speaker_name):
#     sample_rate = 24000
#     if audio_input is None or speaker_name.strip() == "":
#         return gr.update(), "Please upload a file and enter speaker name."

#     try:
#         # Case 1: input is a string filepath
#         if isinstance(audio_input, str):
#             ext = os.path.splitext(audio_input)[1]
#             dest_path = os.path.join(speakers_dir, f"{speaker_name}{ext}")
#             shutil.copy(audio_input, dest_path)

#             # Now convert copied file to 24kHz and 32-bit float WAV
#             data, sr = sf.read(dest_path, dtype='float32')
#             data_resampled = librosa.resample(data.T, orig_sr=sr, target_sr=sample_rate).T
#             sf.write(dest_path, data_resampled, sample_rate, subtype='FLOAT')

#         # Case 2: input is (sample_rate, np.ndarray)
#         elif isinstance(audio_input, tuple) and len(audio_input) == 2:
#             sample_rate_input, audio_array = audio_input
#             dest_path = os.path.join(speakers_dir, f"{speaker_name}.wav")
#             # Save numpy array to wav file using soundfile, converting sample rate if needed
#             if sample_rate_input != sample_rate:
#                 audio_array = librosa.resample(audio_array.T, orig_sr=sample_rate_input, target_sr=sample_rate).T
#             sf.write(dest_path, audio_array, sample_rate, subtype='FLOAT')

#         else:
#             return gr.update(), "Unsupported audio input format."

#         updated_choices = os.listdir(speakers_dir)
#         return gr.update(choices=updated_choices, value=os.path.basename(dest_path)), f"Saved as {dest_path}"

#     except Exception as e:
#         return gr.update(), f"Failed to save uploaded audio: {e}"


# this works but conversion messes up a bit
# def handle_upload(audio_input, text_input_upload):
#     if audio_input is None:
#         return None, "Please upload an audio file."

#     try:
#         # Load audio (works for most formats: mp3, flac, ogg, etc.)
#         waveform, sample_rate = torchaudio.load(audio_input)

#         # Resample to 24kHz if needed
#         if sample_rate != 24000:
#             resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)
#             waveform = resampler(waveform)
#             sample_rate = 24000

#         # Ensure float32 (PyTorch tensors are usually float32 by default)
#         waveform = waveform.to(torch.float32)

#         # Create destination path
#         base_name = os.path.splitext(os.path.basename(audio_input))[0]
#         dest_path = os.path.join(speakers_dir, f"{text_input_upload}.wav")

#         # Write to 32-bit float WAV
#         sf.write(dest_path, waveform.T.numpy(), samplerate=sample_rate, subtype='FLOAT')

#         updated_choices = os.listdir(speakers_dir)
#         return gr.update(choices=updated_choices, value=os.path.basename(dest_path)), f"Saved as {dest_path}"
#         # return dest_path, f"Converted and saved to: {dest_path}"

#     except Exception as e:
#         return None, f"Error processing audio: {str(e)}"


# def handle_upload(audio_input, text_input_upload):
#     if audio_input is None or text_input_upload.strip() == "":
#         return None, "Please upload an audio file and enter a speaker name."

#     try:
#         ext = os.path.splitext(audio_input)[1].lower()
#         dest_path = os.path.join(speakers_dir, f"{text_input_upload}{ext}")

#         waveform, sample_rate = torchaudio.load(audio_input)

#         # Resample to 24kHz if needed
#         if sample_rate != 24000:
#             resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)
#             waveform = resampler(waveform)

#         # Convert to float32 explicitly
#         waveform = waveform.to(torch.float32)

#         # If format supports float32 (e.g. .wav or .flac), use soundfile
#         if ext in [".wav", ".flac"]:
#             import soundfile as sf
#             sf.write(dest_path, waveform.T.numpy(), samplerate=24000, subtype="FLOAT")
#         else:
#             # Use torchaudio for unsupported formats (no float32 guaranteed)
#             torchaudio.save(dest_path, waveform, 24000)

#         updated_choices = os.listdir(speakers_dir)
#         return gr.update(choices=updated_choices, value=os.path.basename(dest_path)), f"Saved as {dest_path}"

#     except Exception as e:
#         return None, f"Error processing audio: {str(e)}"


def handle_upload(audio_input, text_input_upload):
    if audio_input is None or text_input_upload.strip() == "":
        return None, "Please upload an audio file and enter a speaker name."

    try:
        dest_path = os.path.join(speakers_dir, f"{text_input_upload}.wav")

        waveform, sample_rate = torchaudio.load(audio_input)

        if sample_rate != 24000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)
            waveform = resampler(waveform)

        waveform = waveform.to(torch.float32)
        import soundfile as sf
        sf.write(dest_path, waveform.T.numpy(), samplerate=24000, subtype="FLOAT")

        updated_choices = os.listdir(speakers_dir)
        return gr.update(choices=updated_choices, value=os.path.basename(dest_path)), f"Saved as {dest_path}"

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
                # speaker_choices = [f for f in os.listdir(speakers_dir)] # if f.endswith(".wav")
                speakers = gr.Dropdown(choices=[], value="en_sample.wav", label="Choose a speaker")
                # speakers = gr.Dropdown(choices=["en_sample.wav"], label="Choose a speaker")
                btn = gr.Button("Submit")

        # Link upload button
        btn_upload.click(fn=handle_upload, inputs=[file_upload, text_input_upload], outputs=[speakers, upload_status])

        # Link generation button
        btn.click(fn=generate, inputs=[text_input, speakers])

        demo.load(lambda: gr.update(choices=get_speaker_list()), outputs=speakers)
    demo.launch()
