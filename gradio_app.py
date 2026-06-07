import gradio as gr
import torch
import soundfile as sf
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

# Le chemin vers le dossier COMPLET que tu as récupéré de Lightning.ai
PATH_TO_MODEL = "./mon_modele_tts_complet" 

print("Chargement du modèle entraîné personnalisé...")
# Le processeur et le modèle se chargent directement depuis ton dossier local
processor = LFM2AudioProcessor.from_pretrained(PATH_TO_MODEL)
model = LFM2AudioModel.from_pretrained(
    PATH_TO_MODEL, 
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
print("Modèle prêt à l'action en local !")

def generate_japanese_tts(text):
    if not text.strip():
        return None
        
    # On recrée la structure de dialogue attendue par le modèle
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "Perform TTS in Japanese. Use a natural Japanese female voice."}]},
        {"role": "user", "content": [{"type": "text", "text": text}]}
    ]
    
    # Vectorisation du texte
    inputs = processor.apply_chat_template(conversation, return_tensors="pt").to(model.device)
    
    # Inférence (Génération de l'audio)
    with torch.no_grad():
        out_audio = model.generate(**inputs)
        
    # Extraction des données pour Gradio (Fréquence, vagues audio)
    audio_data = out_audio[0].cpu().numpy()
    sampling_rate = processor.sampling_rate
    
    return (sampling_rate, audio_data)

# Interface utilisateur Gradio
demo = gr.Interface(
    fn=generate_japanese_tts,
    inputs=gr.Textbox(lines=3, label="Texte Japonais à synthétiser", placeholder="こんにちは..."),
    outputs=gr.Audio(label="Audio Généré (Voix Spécifique)"),
    title="LiquidAudio LFM 2.5 - TTS Japonais personnalisé",
    description="Entrez du texte en japonais pour générer de l'audio avec le modèle fine-tuné sur Lightning.ai."
)

if __name__ == "__main__":
    demo.launch()