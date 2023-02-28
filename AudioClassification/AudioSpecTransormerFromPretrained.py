import torch
import torchaudio
from transformers import ASTFeatureExtractor
from transformers import AutoModelForAudioClassification

filepath = 'Path to audio file'
feature_extractor = ASTFeatureExtractor(max_length=1024)

audio, sampling_rate = torchaudio.load(filepath)
transform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
audio = transform(audio)

if audio.dim == 2:
    audio = torch.mean(audio, dim=0, keepdim=False).numpy()
else:
    audio = audio.squeeze().numpy()

inputs = feature_extractor(audio, sampling_rate=16000, padding="max_length", return_tensors="pt")
input_values = inputs.input_values


model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
with torch.no_grad():
    outputs = model(input_values)

predicted_class_idx = outputs.logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
