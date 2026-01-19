import os
import torch
import librosa
import torchaudio
import numpy as np
import soundfile as sf
import gradio as gr
import torch.nn.functional as F
from scipy.signal import medfilt
from TTS.api import TTS
from speechbrain.inference import EncoderClassifier
from typing import List, Tuple, Generator, Optional


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SPEAKERS_DIR = "speakers"
    SAMPLES_DIR = os.path.join(SPEAKERS_DIR, "Samples")
    OUTPUTS_DIR = "outputs"
    SAMPLE_RATE = 24000
    HEADER_STR = "--- Model Samples ---"
    DEFAULT_SPEAKER = os.path.join("Samples", "en_sample.wav")
    
    LANGUAGE_MAP = {
        "English": "en", "Spanish": "es", "French": "fr", "German": "de",
        "Italian": "it", "Portuguese": "pt", "Polish": "pl", "Turkish": "tr",
        "Russian": "ru", "Dutch": "nl", "Czech": "cs", "Arabic": "ar",
        "Chinese": "zh-cn", "Japanese": "ja", "Hungarian": "hu",
        "Korean": "ko", "Hindi": "hi"
    }


class SpeakerManager:
    """Handles the filesystem operations for the speaker library."""
    def __init__(self):
        for d in [Config.SPEAKERS_DIR, Config.SAMPLES_DIR, Config.OUTPUTS_DIR]:
            os.makedirs(d, exist_ok=True)

    def get_list(self) -> List[str]:
        main_speakers = sorted([f for f in os.listdir(Config.SPEAKERS_DIR) if f.endswith('.wav')])
        sample_speakers = []
        if os.path.exists(Config.SAMPLES_DIR):
            sample_speakers = sorted([
                os.path.join("Samples", f) for f in os.listdir(Config.SAMPLES_DIR) if f.endswith('.wav')
            ])
        
        final_list = []
        if main_speakers:
            final_list.extend(main_speakers)
        if sample_speakers:
            final_list.append(Config.HEADER_STR)
            final_list.extend(sample_speakers)
        return final_list

    def delete(self, filename: str) -> Tuple[bool, str]:
        if not filename or filename == Config.HEADER_STR:
            return False, "Invalid selection."
        if filename.startswith("Samples"):
            return False, "⚠️ Cannot delete system samples."
        
        path = os.path.join(Config.SPEAKERS_DIR, filename)
        if os.path.exists(path):
            os.remove(path)
            return True, f"🗑️ Deleted: {filename}"
        return False, "File not found."

class AudioEngine:
    """Handles the heavy lifting of audio processing and verification."""
    def __init__(self):
        print(f"Loading Encoder on {Config.DEVICE}...")
        self.spk_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": Config.DEVICE}
        )

    def process_speaker(self, audio_path: str, name: str) -> str:
        save_path = os.path.join(Config.SPEAKERS_DIR, f"{name.strip()}.wav")
        y, sr = librosa.load(audio_path, sr=24000)

        yt, _ = librosa.effects.trim(y, top_db=40)
        
        if len(yt) > 0:
            yt = yt / (np.max(np.abs(yt)) + 1e-9)

        sf.write(save_path, yt, 24000)
        return save_path

    def get_emb(self, p):
        sig, s = torchaudio.load(p)
        if sig.shape[0] > 1: 
            sig = sig.mean(0, keepdim=True)
        
        if s != 24000: 
            sig = torchaudio.transforms.Resample(s, 24000)(sig)
            
        return F.normalize(self.spk_encoder.encode_batch(sig.to(Config.DEVICE)).squeeze(), dim=0)

    def analyze_similarity(self, ref_path, gen_path):
        if not os.path.exists(gen_path): 
            return ""
            
        score = torch.dot(self.get_emb(ref_path), self.get_emb(gen_path)).item()
        pct = max(0, min(100, ((score + 1) / 2) * 100))

        status_map = [(0.85, "💎 Excellent"), (0.75, "🔥 Very Good"), (0.6, "✅ Good"), (0.4, "⚠️ Fair")]
        label = next((text for limit, text in status_map if score >= limit), "❌ Poor")

        return f"""
        <div style='background: #2d2d2d; padding: 15px; border-radius: 8px; color: white;'>
            <h3 style='margin:0;'>Metric Results</h3>
            <p style='font-size: 20px; margin: 5px 0;'>Match: <b>{pct:.1f}%</b></p>
            <p>Quality: {label}</p>
            <small>Cosine Score: {score:.3f}</small>
        </div>
        """

class TTSManager:
    """Manages the XTTS v2 model and inference logic."""
    def __init__(self):
        print(f"Loading XTTS v2 on {Config.DEVICE}...")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(Config.DEVICE)

    def run_inference(self, text, ref_path, lang_code, stream_mode, chunk_size) -> Generator:
        out_path = os.path.join(Config.OUTPUTS_DIR, "latest_gen.wav")
        
        if stream_mode:
            model = self.tts.synthesizer.tts_model
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[ref_path])
            
            chunks = model.inference_stream(
                text, 
                lang_code, 
                gpt_cond_latent, 
                speaker_embedding,
                stream_chunk_size=int(chunk_size),
                enable_text_splitting=True,
                overlap_wav_len=1024,
                speed=1.0,
                repetition_penalty=1.8,
                temperature=0.75
            )
            
            full_audio_for_file = []
            for chunk in chunks:
                chunk_np = chunk.cpu().numpy()
                if len(chunk_np) < 50: continue 
                full_audio_for_file.append(chunk_np)
                yield (24000, chunk_np), "STREAMING"
            
            if full_audio_for_file:
                # Silent padding to ensure playback completion
                silence = np.zeros(int(24000 * 0.3), dtype=np.float32)
                yield (24000, silence), "STREAMING"
                
                final_wav = np.concatenate(full_audio_for_file)
                sf.write(out_path, final_wav, 24000)
                yield out_path, "DONE"
        else:
            wav = self.tts.tts(text=text, speaker_wav=ref_path, language=lang_code, split_sentences=True)
            wav_np = np.array(wav)
            sf.write(out_path, wav_np, 24000)
            yield out_path, "DONE"


class VoiceApp:
    def __init__(self):
        self.speaker_mgr = SpeakerManager()
        self.audio_engine = AudioEngine()
        self.tts_mgr = TTSManager()

    def add_speaker_handler(self, audio_path, name):
        if not audio_path or not name.strip():
            return gr.update(), "Error: Missing info"
        self.audio_engine.process_speaker(audio_path, name)
        choices = self.speaker_mgr.get_list()
        return gr.update(choices=choices, value=f"{name.strip()}.wav"), f"✅ Added: {name}"

    def delete_speaker_handler(self, speaker_file):
        success, msg = self.speaker_mgr.delete(speaker_file)
        choices = self.speaker_mgr.get_list()
        new_val = next((s for s in choices if s != Config.HEADER_STR), None)
        return gr.update(choices=choices, value=new_val), msg, gr.update(visible=False)

    def run_generation_handler(self, text, speaker_file, lang_name, stream_mode, chunk_size):
        if not text or not speaker_file or speaker_file == Config.HEADER_STR:
            yield None, "Select speaker and enter text."
            return

        lang_code = Config.LANGUAGE_MAP.get(lang_name, "en")
        ref_path = os.path.join(Config.SPEAKERS_DIR, speaker_file)
        out_path = os.path.join(Config.OUTPUTS_DIR, "latest_gen.wav")

        gen = self.tts_mgr.run_inference(text, ref_path, lang_code, stream_mode, chunk_size)
        
        for data, state in gen:
            if state == "DONE":
                analysis_html = self.audio_engine.analyze_similarity(ref_path, out_path)

                if stream_mode:
                    yield None, analysis_html
                else:
                    yield data, analysis_html
            elif state == "STREAMING":
                yield data, gr.update()
            else:
                yield data, "Synthesizing..."

    def build_ui(self):
        with gr.Blocks() as demo:
            gr.HTML("<h1 style='text-align: center;'>🎙️ Voice Clone & TTS </h1>")

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
                        current_choices = self.speaker_mgr.get_list()
                        default_val = Config.DEFAULT_SPEAKER if Config.DEFAULT_SPEAKER in current_choices else (current_choices[0] if current_choices else None)
                        spk_select = gr.Dropdown(
                            choices=current_choices, 
                            label="Select Active Speaker",
                            value=default_val,
                            scale=2
                        )
                        lang_select = gr.Dropdown(
                            choices=list(Config.LANGUAGE_MAP.keys()),
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
                        chunk_slider = gr.Slider(minimum=20, maximum=400, step=10, value=150, label="Chunk Size")
                    
                    gen_btn = gr.Button("🚀 Generate & Analyze", variant="primary", size="lg")

                with gr.Column(scale=1, variant="panel"):
                    gr.Markdown("### 🔊 Result")
                    audio_out = gr.Audio(label="Generated Audio", interactive=False, autoplay=True, streaming=True)
                    stats_out = gr.HTML()

            spk_select.focus(lambda: gr.update(choices=self.speaker_mgr.get_list()), outputs=spk_select)
            add_btn.click(self.add_speaker_handler, [audio_in, name_in], [spk_select, status_msg])
            
            delete_btn.click(lambda: gr.update(visible=True), None, confirm_row)
            confirm_no.click(lambda: gr.update(visible=False), None, confirm_row)
            confirm_yes.click(self.delete_speaker_handler, [spk_select], [spk_select, status_msg, confirm_row])
            
            gen_btn.click(self.run_generation_handler, [txt_in, spk_select, lang_select, stream_toggle, chunk_slider], [audio_out, stats_out])

        return demo

if __name__ == "__main__":
    app = VoiceApp()
    app.build_ui().launch(
        # share=True,
        theme=gr.themes.Default(primary_hue="orange", secondary_hue="gray")
    )