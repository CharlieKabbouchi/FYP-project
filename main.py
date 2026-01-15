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

print(f"Initializing Models on {device}...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
# tts = TTS("tts_models/multilingual/multi-dataset/bark").to(device)

spk_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)



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

    current_speakers = sorted(os.listdir(speakers_dir))
    return gr.update(choices=current_speakers, value=f"{name.strip()}.wav"), f"✅ Added: {name}"

def run_generation(text, speaker_file, stream_mode, chunk_size, progress=gr.Progress()):
    if not text or not speaker_file:
        return None, "Select speaker and enter text."

    ref_path = os.path.join(speakers_dir, speaker_file)
    out_path = os.path.join(outputs_dir, "latest_gen.wav")

    if stream_mode:
        try:
            model = tts.synthesizer.tts_model
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[ref_path])
            

            chunks = model.inference_stream(
                text,
                "en",
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
            
            # Save final concatenated file
            if full_audio_for_file:
                final_wav = np.concatenate(full_audio_for_file)
                sf.write(out_path, final_wav, 24000)
                
        except Exception as e:
            yield None, f"Streaming error: {str(e)}"
            return
    else:
        # Generate
        progress(0.2, desc="Synthesizing...")
        wav = tts.tts(text=text, speaker_wav=ref_path, language="en", split_sentences=True)
        sf.write(out_path, wav, 24000)

        # Similarity Analysis
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
    with gr.Blocks(theme=gr.themes.Default(primary_hue="orange", secondary_hue="gray")) as demo:
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
                spk_select = gr.Dropdown(
                    choices=sorted(os.listdir(speakers_dir)), 
                    label="Select Active Speaker",
                    value=sorted(os.listdir(speakers_dir))[0] if os.listdir(speakers_dir) else None
                )
                txt_in = gr.Textbox(label="Text to Speech", lines=8, placeholder="Enter text here...")

                with gr.Row():
                    stream_toggle = gr.Checkbox(label="Real-time Streaming Mode", value=False)
                    # Increased buffer size range
                    chunk_slider = gr.Slider(minimum=50, maximum=500, step=10, value=125, label="Buffer Size (Tokens)")
                
                
                gen_btn = gr.Button("🚀 Generate & Analyze", variant="primary", size="lg")

            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 🔊 Result")
                audio_out = gr.Audio(label="Generated Audio", interactive=False, autoplay=True, streaming=True)
                stats_out = gr.HTML()

        add_btn.click(process_speaker, [audio_in, name_in], [spk_select, status_msg])
        gen_btn.click(run_generation, [txt_in, spk_select, stream_toggle, chunk_slider], [audio_out, stats_out])

    demo.launch(share=True)
    # demo.launch(
    #     server_name="0.0.0.0",
    #     server_port=7860,
    # )


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