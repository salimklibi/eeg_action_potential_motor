import streamlit as st
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import pandas as pd
from pathlib import Path

@st.cache_data
def load_eeg_action_demo_data():
    """Dataset simulé Potentiel d'Action: Cz/Fz 1s pré-mouvement vs repos."""
    sfreq = 256  # Hz
    n_epochs = 500
    duration = 2.0  # -1s à +1s (t=0 mouvement)
    
    data = np.zeros((n_epochs, 1, int(duration * sfreq)))  # 1 canal Cz
    labels = np.random.choice([0, 1], n_epochs, p=[0.5, 0.5])  # 0=repos, 1=préparation
    
    for i in range(n_epochs):
        t = np.linspace(-1, 1, int(duration * sfreq))
        if labels[i] == 1:  # Potentiel d'action: negativity ramp -500ms/-100ms
            action_signal = -5 * np.exp(-((t + 0.3)/0.2)**2) * (t > -0.6)  # Gaussienne négative
            noise = np.random.normal(0, 2, len(t))
            data[i, 0, :] = action_signal + noise
        else:
            data[i, 0, :] = np.random.normal(0, 2, len(t))
    
    info = mne.create_info(ch_names=['Cz'], sfreq=sfreq, ch_types='eeg')
    raw = mne.EpochsArray(data, info, tmin=-1.0)
    return raw, labels

def eeg_preprocessing(raw, l_freq=1, h_freq=45, notch_freq=50):
    """Prétraitement EEG standard neurosciences."""
    raw.notch_filter(np.arange(notch_freq-2, notch_freq+3, 2))
    raw.filter(l_freq, h_freq, fir_design='firwin')
    ica = mne.preprocessing.ICA(n_components=1, random_state=42)
    ica.fit(raw)
    raw.set_eeg_reference('average', projection=True)
    return raw

def extract_action_features(raw, labels):
    """Features temporelles/spatiales Potentiel d'Action préparation motrice."""
    epochs = raw.copy().crop(tmin=-0.8, tmax=-0.05).get_data()  # Fenêtre [-800,-50ms]
    
    features = []
    for epoch in epochs:
        early_action = np.mean(epoch[:, raw.times > -0.6])   # Phase précoce
        late_action = np.mean(epoch[:, raw.times > -0.3])    # Phase tardive
        slope = (late_action - early_action) / 0.25          # Pente activation
        power_alpha = np.mean(np.abs(np.fft.rfft(epoch[:, raw.times > -0.1], n=64))**2)
        features.append([early_action, late_action, slope, power_alpha, np.std(epoch)])
    
    return np.array(features)

st.title(" EEG Action Potential Detector - Préparation Motrice")
st.markdown("**Potentiel d'Action EEG en Cz: Intention motrice détectée ~800ms AVANT mouvement.** BCI/Neurosciences.")

raw, true_labels = load_eeg_action_demo_data()
st.info(f" {len(raw)} epochs Cz 256Hz (-1s/+1s), {np.sum(true_labels)} Potentiels+.")

if st.button(" Prétraitement EEG"):
    raw_clean = eeg_preprocessing(raw.copy())
    st.success("Prétraitement OK!")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
action_epochs = raw[true_labels == 1].average()
rest_epochs = raw[true_labels == 0].average()
action_epochs.plot(axes=ax1, show=False)
ax1.set_title("Grand Moyenne: Potentiel Action (Rouge) vs Repos")
rest_epochs.plot(axes=ax2, show=False)
ax2.set_title("Cz: Negativity pré-mouvement")
st.pyplot(fig)

if st.button(" Détecteur Potentiel d'Action"):
    features = extract_action_features(raw_clean, true_labels)
    X_train, X_test, y_train, y_test = train_test_split(features, true_labels, test_size=0.3)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{report['accuracy']:.1%}")
    col2.metric("Precision Action+", f"{report['1']['precision']:.1%}")
    col3.metric("Recall Action+", f"{report['1']['recall']:.1%}")
    
    importance_df = pd.DataFrame({
        'Feature': ['Early_Action', 'Late_Action', 'Slope', 'Alpha_Power', 'Variance'],
        'Importance': model.feature_importances_
    })
    fig_imp = px.bar(importance_df, x='Feature', y='Importance')
    st.plotly_chart(fig_imp)

st.markdown("""
**Potentiel d'Action Préparation Motrice:**
- **Phase précoce [-800ms]**: Intention consciente
- **Phase tardive [-300ms]**: Activation finale
- **Cz negativity ~5μV**: Détectable BCI
""")
