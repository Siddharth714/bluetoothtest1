import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import torchvision.transforms as transforms
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

one_hot_mapping = {
    0: [1, 0, 0, 0],  # Healthy
    1: [0, 1, 0, 0],  # Very Mild Demented
    2: [0, 0, 1, 0],  # Mild Demented
    3: [0, 0, 0, 1]   # Moderate Demented
}

selected_features = [
    "total_time", "num_of_pendown", "mean_acc_in_air", "max_y_extension",
    "disp_index", "mean_gmrt", "gmrt_on_paper", "max_x_extension",
    "paper_time", "air_time", "mean_acc_on_paper", "mean_jerk_in_air",
    "mean_jerk_on_paper", "pressure_var", "mean_speed_on_paper",
    "mean_speed_in_air", "pressure_mean", "gmrt_in_air"
]


class EEGModel(nn.Module):
    def __init__(self, eeg_file):
        super(EEGModel, self).__init__()
        self.eeg_file = eeg_file
        self.eeg_mapping = {
            "Normal Cognition Pattern": [1, 0, 0, 0],
            "Mild Early Stage Alzheimer's Detected": [0, 1, 0, 0],
            "Moderate Confidence: Alzheimer's Disease": [0, 0, 1, 0],
            "High Confidence: Severe Alzheimer's Disease": [0, 0, 0, 1]
        }
        self.eeg_inverse_mapping = {
            0: "Normal Cognition Pattern",
            1: "Mild Early Stage Alzheimer's Detected",
            2: "Moderate Confidence: Alzheimer's Disease",
            3: "High Confidence: Severe Alzheimer's Disease"
        }

    def forward(self, dummy_input=None):
        raw = mne.io.read_raw_eeglab(self.eeg_file, preload=True, verbose=False)
        if 'A1' in raw.ch_names and 'A2' in raw.ch_names:
            raw.set_eeg_reference(ref_channels=['A1', 'A2'])
        else:
            raw.set_eeg_reference('average')
        raw.filter(0.5, 45, method='iir', iir_params={'order': 4, 'ftype': 'butter'})
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage)
        raw_ica = raw.copy().filter(1, 100, method='iir', iir_params={'order': 4, 'ftype': 'butter'})
        ica = ICA(n_components=18, method='infomax', fit_params=dict(extended=True), random_state=97)
        ica.fit(raw_ica)
        ic_labels = label_components(raw_ica, ica, method="iclabel")
        labels = ic_labels["labels"]
        exclude_idx = [idx for idx, label in enumerate(labels) if label in ["eye blink", "muscle artifact", "eye movement"]]
        ica.exclude = exclude_idx
        raw_clean = ica.apply(raw.copy())
        epochs = mne.make_fixed_length_epochs(raw_clean, duration=2, overlap=1, preload=True)
        psd_obj = epochs.compute_psd(method='welch', fmin=0.5, fmax=45, n_fft=1000, n_overlap=500)
        psds = psd_obj.get_data()
        freqs = psd_obj.freqs
        freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 45)
        }
        band_power = {}
        for band, (fmin, fmax) in freq_bands.items():
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            band_power[band] = psds[:, :, freq_mask].mean(axis=(0, 2))
        df_power = pd.DataFrame(band_power, index=epochs.ch_names)
        df_power['theta/alpha'] = df_power['theta'] / df_power['alpha']
        df_power['delta/alpha'] = df_power['delta'] / df_power['alpha']
        df_power['beta/theta'] = df_power['beta'] / df_power['theta']
        df_power['gamma/theta'] = df_power['gamma'] / df_power['theta']

        def classify_severity(row):
            cn_theta_alpha_mean = 1.2
            cn_theta_alpha_std = 0.3
            cn_delta_alpha_mean = 0.8
            cn_delta_alpha_std = 0.2
            z_theta_alpha = (row['theta/alpha'] - cn_theta_alpha_mean) / cn_theta_alpha_std
            z_delta_alpha = (row['delta/alpha'] - cn_delta_alpha_mean) / cn_delta_alpha_std
            if z_theta_alpha > 3.5 or z_delta_alpha > 3.5:
                return "High Confidence: Severe Alzheimer's Disease"
            elif z_theta_alpha > 2.5 or z_delta_alpha > 2.5:
                return "Moderate Confidence: Alzheimer's Disease"
            elif z_theta_alpha > 1.5 or z_delta_alpha > 1.5:
                return "Mild Early Stage Alzheimer's Detected"
            else:
                return "Normal Cognition Pattern"
        df_power['Severity'] = df_power.apply(classify_severity, axis=1)

        def get_final_conclusion(df):
            posterior_channels = ['O1', 'O2', 'P3', 'P4', 'T5', 'T6']
            counts = {"Normal Cognition Pattern": 0, 
                      "Mild Early Stage Alzheimer's Detected": 0, 
                      "Moderate Confidence: Alzheimer's Disease": 0, 
                      "High Confidence: Severe Alzheimer's Disease": 0}
            for ch in df.index:
                sev = df.loc[ch, 'Severity']
                weight = 1.5 if ch in posterior_channels else 1.0
                counts[sev] += weight
            total = sum(counts.values())
            if counts["High Confidence: Severe Alzheimer's Disease"] / total > 0.3:
                return "High Confidence: Severe Alzheimer's Disease"
            elif counts["Moderate Confidence: Alzheimer's Disease"] / total > 0.25:
                return "Moderate Confidence: Alzheimer's Disease"
            elif counts["Mild Early Stage Alzheimer's Detected"] / total > 0.2:
                return "Mild Early Stage Alzheimer's Detected"
            else:
                return "Normal Cognition Pattern"
        final_verdict = get_final_conclusion(df_power)
        verdict_vector = self.eeg_mapping.get(final_verdict, self.eeg_mapping["Normal Cognition Pattern"])
        return torch.tensor(verdict_vector, dtype=torch.float32).unsqueeze(0)


class HandwritingModel(nn.Module):
    def __init__(self, model_path):
        super(HandwritingModel, self).__init__()
        self.combined = joblib.load(model_path)
        self.model = self.combined["model"]
        self.scalar = self.combined["scaler"]
        self.selected_features = self.combined["features"]
        self.one_hot_mapping = one_hot_mapping
        self.handwriting_inverse_mapping = {
            3: "Healthy",           # from [0,0,0,1]
            1: "Very Mild Demented",  # from [0,1,0,0]
            2: "Mild Demented",       # from [0,0,1,0]
            0: "Moderate Demented"    # from [1,0,0,0]
        }

    def forward(self, csv_path):
       
        df = pd.read_csv(csv_path)
        df = df[self.selected_features]
        input_scaled = self.scalar.transform(df)
        pred_labels = self.model.predict(input_scaled)[0]
        one_hot_preds = [self.one_hot_mapping[pred_labels]]
        print("Handwriting branch output (one-hot):", one_hot_preds)
        return torch.tensor(one_hot_preds, dtype=torch.float32)


class MRIModel(nn.Module):
    def __init__(self, model_path):
        super(MRIModel, self).__init__()
        loaded_obj = joblib.load(model_path)
        if isinstance(loaded_obj, dict):
            self.model = loaded_obj["model"]
            self.scaler = loaded_obj.get("scaler", None)
        else:
            self.model = loaded_obj
            self.scaler = None
        self.one_hot_mapping = {
            0: [0, 1, 0, 0],  # Healthy
            1: [1, 0, 0, 0],  # Very Mild Demented
            2: [0, 0, 0, 1],  # Mild Demented
            3: [0, 0, 1, 0]   # Moderate Demented
        }
        self.inverse_mapping = {
            1: "Healthy",
            0: "Very Mild Demented",
            3: "Mild Demented",
            2: "Moderate Demented"
        }
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def forward(self, image_path):
        
        img = Image.open(image_path).convert("RGB")
        mri_tensor = self.transform(img).unsqueeze(0)  
        mri_tensor = mri_tensor.permute(0, 2, 3, 1)      
        x_np = mri_tensor.detach().cpu().numpy() / 255.0
        pred_probs = self.model.predict(x_np)  
        pred_indices = np.argmax(pred_probs, axis=1)
        one_hot_preds = [self.one_hot_mapping[int(idx)] for idx in pred_indices]
        print("MRI branch output (one-hot):", one_hot_preds)
        return torch.tensor(one_hot_preds, dtype=torch.float32)


class MultiModalPredictor(nn.Module):
    def __init__(self, eeg_model, handwriting_model, mri_model, final_output_dim=4):
        super(MultiModalPredictor, self).__init__()
        self.eeg_model = eeg_model
        self.handwriting_model = handwriting_model
        self.mri_model = mri_model

    def forward(self, handwriting_input, mri_input, eeg_dummy_input=None):
        eeg_out = self.eeg_model(eeg_dummy_input)            
        handwriting_out = self.handwriting_model(handwriting_input) 
        if eeg_out.size(0) != handwriting_out.size(0):
            eeg_out = eeg_out.expand(handwriting_out.size(0), -1)

        eeg_idx = torch.argmax(eeg_out[0]).item()
        handwriting_idx = torch.argmax(handwriting_out[0]).item()
        eeg_label = self.eeg_model.eeg_inverse_mapping[eeg_idx]
        handwriting_label = self.handwriting_model.handwriting_inverse_mapping[handwriting_idx]
        print("EEG predicted label:", eeg_label)
        print("Handwriting predicted label:", handwriting_label)

        if (eeg_idx == 3) or (handwriting_idx == 0):
            if mri_input is None:
                reason = ("MRI is required but not provided, so averaging EEG and Handwriting outputs.")
                final_logits = (eeg_out + handwriting_out) / 2
            else:
                mri_out = self.mri_model(mri_input)
                if mri_out.size(0) != handwriting_out.size(0):
                    mri_out = mri_out.expand(handwriting_out.size(0), -1)
                # Average the outputs from all three branches.
                final_logits = (eeg_out + handwriting_out + mri_out) / 3
                reason = ("MRI branch was included because either EEG predicted 'High Confidence: Severe Alzheimer's Disease' "
                          "or Handwriting predicted 'Moderate Demented'.")
        else:
           
            final_logits = (eeg_out + handwriting_out) / 2
            reason = ("MRI branch was not used because EEG predicted '{}' and Handwriting predicted '{}'."
                      .format(eeg_label, handwriting_label))
        return final_logits, reason


if __name__ == "__main__":

    eeg_file_path = input("Enter the path to the EEG set file: ")
    handwriting_model_path = r"C:\Users\Siddharth\Desktop\Projects\Alzheimer's Detection\combined_model_scaler_smoteenn_randomized.pkl"
    mri_model_path = r"C:\Users\Siddharth\Desktop\Projects\Alzheimer's Detection\model.pkl"

    eeg_model = EEGModel(eeg_file=eeg_file_path)
    handwriting_model = HandwritingModel(model_path=handwriting_model_path)
    mri_model = MRIModel(model_path=mri_model_path)

    # Instantiate the multimodal predictor.
    multi_modal_model = MultiModalPredictor(eeg_model, handwriting_model, mri_model, final_output_dim=4)

    handwriting_csv_path = input("Enter the path to the handwriting CSV file: ")
    mri_input = None
    eeg_dummy_input = None

    # Initial prediction without MRI.
    initial_logits, initial_reason = multi_modal_model(handwriting_csv_path, mri_input, eeg_dummy_input)
    initial_probs = torch.softmax(initial_logits, dim=1)
    initial_pred_idx = torch.argmax(initial_probs, dim=1).item()

    # EEG inverse mapping for final interpretation.
    final_mapping = eeg_model.eeg_inverse_mapping
    initial_pred_class = final_mapping.get(initial_pred_idx, "Unknown")
    print("Initial Predicted class:", initial_pred_class)
    print("Reason:", initial_reason)

    # If severe prediction requires MRI, then prompt for MRI input.
    if (initial_pred_class == "High Confidence: Severe Alzheimer's Disease") or ("MRI is required" in initial_reason):
        print("MRI is required to confirm the diagnosis.")
        mri_image_path = input("Enter the path to the MRI image file: ")
        mri_input = mri_image_path
        final_logits, final_reason = multi_modal_model(handwriting_csv_path, mri_input, eeg_dummy_input)
        final_probs = torch.softmax(final_logits, dim=1)
        final_pred_idx = torch.argmax(final_probs, dim=1).item()
        final_pred_class = final_mapping.get(final_pred_idx, "Unknown")
        print("Final Predicted class:", final_pred_class)
        print("Reason:", final_reason)
    else:
        print("Final Predicted class:", initial_pred_class)
        print("Reason:", initial_reason)
