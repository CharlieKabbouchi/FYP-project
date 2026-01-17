from TTS.api import TTS
import torch
import numpy as np
import gradio as gr
import os
import librosa
import torchaudio
import soundfile as sf
from scipy.signal import medfilt
from speechbrain.inference import EncoderClassifier
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
speakers_dir = "speakers"
outputs_dir = "outputs"
for d in [speakers_dir, outputs_dir]: os.makedirs(d, exist_ok=True)

LANGUAGE_MAP = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Hungarian": "hu",
    "Korean": "ko",
    "Hindi": "hi"
}


print(f"Initializing Models on {device}...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

spk_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)


def get_speaker_list():
    """Helper to get current files from the speakers directory with grouping for Samples."""
    main_speakers = sorted([f for f in os.listdir(speakers_dir) if f.endswith('.wav')])
    
    samples_path = os.path.join(speakers_dir, "Samples")
    sample_speakers = []
    if os.path.exists(samples_path):
        sample_speakers = sorted([os.path.join("Samples", f) for f in os.listdir(samples_path) if f.endswith('.wav')])
    
    final_list = []
    if main_speakers:
        final_list.extend(main_speakers)
    
    if sample_speakers:
        final_list.append("--- Model Samples ---")
        final_list.extend(sample_speakers)
        
    return final_list


def refresh_speakers():
    """Function to refresh the dropdown choices dynamically."""
    choices = get_speaker_list()
    return gr.update(choices=choices)


def process_speaker(audio_path, name):
    if not audio_path or not name.strip():
        return gr.update(), "Error: Missing info"

    save_path = os.path.join(speakers_dir, f"{name.strip()}.wav")

    # Load and standardize
    y, sr = librosa.load(audio_path, sr=24000)

    # Quick Noise Reduction
    S_full, phase = librosa.magphase(librosa.stft(y))
    noise_power = np.mean(S_full[:, :int(24000 * 0.1)], axis=1)
    mask = medfilt((S_full > noise_power[:, None]).astype(float), kernel_size=(1, 5))
    y_clean = librosa.istft(S_full * mask * phase)

    # Trim Silence
    yt, _ = librosa.effects.trim(y_clean, top_db=35)

    sf.write(save_path, yt, 24000)

    current_speakers = get_speaker_list()
    return gr.update(choices=current_speakers, value=f"{name.strip()}.wav"), f"✅ Added: {name}"


def delete_speaker(speaker_file):
    if not speaker_file or speaker_file == "--- Model Samples ---":
        return gr.update(), "Invalid selection for deletion.", gr.update(visible=False)
    
    # Check if the speaker is inside the Samples directory
    if speaker_file.startswith("Samples"):
        return gr.update(), "⚠️ Cannot delete system samples.", gr.update(visible=False)
    
    file_path = os.path.join(speakers_dir, speaker_file)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            current_speakers = get_speaker_list()
            new_val = next((s for s in current_speakers if s != "--- Model Samples ---"), None)
            return gr.update(choices=current_speakers, value=new_val), f"🗑️ Deleted: {speaker_file}", gr.update(visible=False)
        else:
            return gr.update(), "File not found.", gr.update(visible=False)
    except Exception as e:
        return gr.update(), f"Error deleting: {str(e)}", gr.update(visible=False)


def run_generation(text, speaker_file, language_name, stream_mode, chunk_size, progress=gr.Progress()):
    if not text or not speaker_file:
        return None, "Select speaker and enter text."

    lang_code = LANGUAGE_MAP.get(language_name, "en")
    ref_path = os.path.join(speakers_dir, speaker_file)
    out_path = os.path.join(outputs_dir, "latest_gen.wav")

    if stream_mode:
        try:
            model = tts.synthesizer.tts_model
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[ref_path])
            
            chunks = model.inference_stream(
                text,
                lang_code,
                gpt_cond_latent,
                speaker_embedding,
                stream_chunk_size=int(chunk_size),
                enable_text_splitting=True
            )
            
            full_audio_for_file = []
            for chunk in chunks:
                chunk_np = chunk.cpu().numpy()
                full_audio_for_file.append(chunk_np)
                yield (24000, chunk_np), "Streaming..."
            
            if full_audio_for_file:
                final_wav = np.concatenate(full_audio_for_file)
                sf.write(out_path, final_wav, 24000)
                
        except Exception as e:
            yield None, f"Streaming error: {str(e)}"
            return
    else:
        progress(0.2, desc="Synthesizing...")
        wav = tts.tts(text=text, speaker_wav=ref_path, language=lang_code, split_sentences=True)
        sf.write(out_path, wav, 24000)

    progress(0.8, desc="Analyzing Voiceprint...")

    def get_emb(p):
        sig, s = torchaudio.load(p)
        if sig.shape[0] > 1: sig = sig.mean(0, keepdim=True)
        if s != 16000: sig = torchaudio.transforms.Resample(s, 24000)(sig)
        return F.normalize(spk_encoder.encode_batch(sig.to(device)).squeeze(), dim=0)

    score = torch.dot(get_emb(ref_path), get_emb(out_path)).item()
    pct = max(0, min(100, ((score + 1) / 2) * 100))

    status_map = [(0.85, "💎 Excellent"), (0.75, "🔥 Very Good"), (0.6, "✅ Good"), (0.4, "⚠️ Fair")]
    label = next((text for limit, text in status_map if score >= limit), "❌ Poor")

    result_html = f"""
    <div style='background: #2d2d2d; padding: 15px; border-radius: 8px; color: white;'>
        <h3 style='margin:0;'>Metric Results</h3>
        <p style='font-size: 20px; margin: 5px 0;'>Match: <b>{pct:.1f}%</b></p>
        <p>Quality: {label}</p>
        <small>Cosine Score: {score:.3f}</small>
    </div>
    """
    
    yield (None if stream_mode else out_path), result_html


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.HTML("<h1 style='text-align: center;'>🎙️ Voice clone & TTS </h1>")

        with gr.Row():
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 👤 Speaker Source")
                audio_in = gr.Audio(label="Record/Upload", type="filepath")
                name_in = gr.Textbox(label="Speaker Name", placeholder="Enter name...")
                add_btn = gr.Button("Add to Library", variant="secondary")
                

                status_msg = gr.Markdown("---")

            with gr.Column(scale=2, variant="compact"):
                gr.Markdown("### ✍️ Synthesis")
                
                with gr.Row():
                    current_choices = get_speaker_list()
                    spk_select = gr.Dropdown(
                        choices=current_choices, 
                        label="Select Active Speaker",
                        value=current_choices[0] if current_choices else None,
                        scale=2
                    )
                    lang_select = gr.Dropdown(
                        choices=list(LANGUAGE_MAP.keys()),
                        label="Language",
                        value="English",
                        scale=1
                    )
                with gr.Accordion("Deletion Block", open=False):
                        delete_btn = gr.Button("🗑️ Delete Selected Speaker", variant="stop")
                        with gr.Row(visible=False) as confirm_row:
                            confirm_yes = gr.Button("Confirm Delete", variant="stop", size="sm")
                            confirm_no = gr.Button("Cancel", size="sm")
                txt_in = gr.Textbox(label="Text to Speech", lines=8, placeholder="Enter text here...")

                with gr.Row():
                    stream_toggle = gr.Checkbox(label="Real-time Streaming Mode", value=False)
                    chunk_slider = gr.Slider(minimum=50, maximum=500, step=10, value=125, label="Buffer Size (Tokens)")
                
                gen_btn = gr.Button("🚀 Generate & Analyze", variant="primary", size="lg")

            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 🔊 Result")
                audio_out = gr.Audio(label="Generated Audio", interactive=False, autoplay=True, streaming=True)
                stats_out = gr.HTML()

        spk_select.focus(fn=refresh_speakers, inputs=None, outputs=spk_select)

        add_btn.click(process_speaker, [audio_in, name_in], [spk_select, status_msg])
        
        delete_btn.click(lambda: gr.update(visible=True), None, confirm_row)
        confirm_no.click(lambda: gr.update(visible=False), None, confirm_row)
        confirm_yes.click(fn=delete_speaker, inputs=[spk_select], outputs=[spk_select, status_msg, confirm_row])
        
        gen_btn.click(run_generation, [txt_in, spk_select, lang_select, stream_toggle, chunk_slider], [audio_out, stats_out])

    demo.launch(
        # share=True,
        theme=gr.themes.Default(primary_hue="orange", secondary_hue="gray")
    )
# def clean_backgroundnoise_of_audio(input_path, output_path):
#     # this is a simple version to clean background noise (some may still exist)

#     y, sr = librosa.load(input_path, sr=None)
#     S_full, phase = librosa.magphase(librosa.stft(y))
#     noise_power = np.mean(S_full[:, :int(sr * 0.1)], axis=1)
#     mask = S_full > noise_power[:, None]
#     mask = mask.astype(float)
#     mask = medfilt(mask, kernel_size=(1, 5))
#     S_clean = S_full * mask
#     y_clean = librosa.istft(S_clean * phase)
#     sf.write(output_path, y_clean, sr, format='WAV', subtype='FLOAT')


# def clean_silence_from_audio(input_path, output_path):
#     y, sr = librosa.load(input_path, sr=None)

#     # Detect non-silent intervals and put them in a single arr
#     intervals = librosa.effects.split(y, top_db=40)
#     y_trimmed = np.concatenate([y[start:end] for start, end in intervals])
#     sf.write(output_path, y_trimmed, sr, format='WAV', subtype='FLOAT')