import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Configuration de la page
st.set_page_config(
    page_title="Predicteur du nombre ideal d enfants",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS personnalise
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+Pro:wght@300;400;600&display=swap');
    
    .main {
        background: linear-gradient(135deg, #FDF8F3 0%, #F5EDE4 100%);
    }
    
    h1 {
        font-family: 'Playfair Display', serif !important;
        color: #2C1810 !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
    }
    
    h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #4A2C17 !important;
        font-weight: 600 !important;
    }
    
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-family: 'Source Sans Pro', sans-serif !important;
        color: #5C3D2E !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #8B4513 0%, #A0522D 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(139, 69, 19, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(139, 69, 19, 0.4) !important;
    }
    
    .stButton > button:disabled {
        background: #C4B5A5 !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
    }
    
    .result-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #FAF6F1 100%);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid #E8DDD0;
        box-shadow: 0 8px 32px rgba(44, 24, 16, 0.08);
        margin-top: 2rem;
    }
    
    .result-number {
        font-family: 'Playfair Display', serif;
        font-size: 4rem;
        font-weight: 700;
        color: #8B4513;
        line-height: 1;
    }
    
    .result-label {
        font-family: 'Source Sans Pro', sans-serif;
        color: #6B5B4F;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #D4A574 50%, transparent 100%);
        margin: 2rem 0;
        border: none;
    }
    
    .info-box {
        background: #FAF6F1;
        border-left: 4px solid #D4A574;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #FFF8F0;
        border-left: 4px solid #C9A227;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .stSelectbox > div > div {
        border-radius: 10px !important;
        border-color: #D4C4B0 !important;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px !important;
        border-color: #D4C4B0 !important;
    }
    
    .footer {
        text-align: center;
        color: #9E8E7E;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E8DDD0;
    }
</style>
""", unsafe_allow_html=True)

# Chargement du modele
@st.cache_resource
def charger_modele():
    chemin_modele = "modele_ml_final.pkl"
    if os.path.exists(chemin_modele):
        return joblib.load(chemin_modele)
    else:
        return None

modele = charger_modele()

# Header
st.markdown("""
<h1 style="text-align: center; margin-bottom: 0.2rem;">Predicteur du nombre ideal d enfants</h1>
<p style="text-align: center; color: #6B5B4F; font-family: 'Source Sans Pro', sans-serif; font-size: 1.1rem; margin-bottom: 2rem;">
Outil d aide a la decision base sur l analyse de donnees demographiques
</p>
""", unsafe_allow_html=True)

if modele is None:
    st.markdown("""
    <div class="warning-box">
        <strong>Attention :</strong> Le modele n a pas ete trouve. 
        Veuillez executer le notebook d entrainement pour generer <code>modele_ml_final.pkl</code>.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Formulaire principal
st.markdown('<h3 style="color: #4A2C17; margin-bottom: 1rem;">Renseignez les informations</h3>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    Tous les champs sont obligatoires. Veuillez selectionner une valeur pour chaque critere afin d obtenir une prediction fiable.
</div>
""", unsafe_allow_html=True)

# Organisation en colonnes
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p style="color: #8B4513; font-weight: 600; font-size: 1.05rem; margin-bottom: 0.8rem;">Situation geographique et sociale</p>', unsafe_allow_html=True)
    
    region = st.selectbox(
        "Region de residence",
        ["Adamaoua", "Centre", "Douala", "Est", "Extrême-Nord", "Littoral", "Nord", "Nord-Ouest", "Ouest", "Sud", "Sud-Ouest", "Yaoundé"],
        index=None,
        placeholder="Selectionner une region..."
    )
    
    richesse = st.selectbox(
        "Niveau de richesse du menage",
        ["Plus pauvre", "Pauvre", "Moyen", "Riche", "Plus riche"],
        index=None,
        placeholder="Selectionner un niveau..."
    )
    
    proprietaire = st.selectbox(
        "Proprietaire de terres agricoles",
        ["Non", "Oui"],
        index=None,
        placeholder="Selectionner..."
    )

with col2:
    st.markdown('<p style="color: #8B4513; font-weight: 600; font-size: 1.05rem; margin-bottom: 0.8rem;">Education et religion</p>', unsafe_allow_html=True)
    
    edu_femme = st.selectbox(
        "Niveau d education de la femme",
        ["Aucune", "Primaire", "Secondaire", "Supérieur"],
        index=None,
        placeholder="Selectionner un niveau..."
    )
    
    edu_mari = st.selectbox(
        "Niveau d education du conjoint",
        ["Aucune", "Primaire", "Secondaire", "Supérieur"],
        index=None,
        placeholder="Selectionner un niveau..."
    )
    
    religion = st.selectbox(
        "Religion",
        ["Chrétienne", "Musulmane", "Autre / Sans religion"],
        index=None,
        placeholder="Selectionner une religion..."
    )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown('<p style="color: #8B4513; font-weight: 600; font-size: 1.05rem; margin-bottom: 0.8rem;">Acces aux informations et sante</p>', unsafe_allow_html=True)
    
    besoin_pf = st.selectbox(
        "Besoin non satisfait en planification familiale",
        ["Non expose", "Expose, besoin satisfait", "Expose, besoin non satisfait"],
        index=None,
        placeholder="Selectionner..."
    )
    
    tv = st.selectbox(
        "Frequence de visionnage de la television",
        ["Pas du tout", "Parfois", "Souvent"],
        index=None,
        placeholder="Selectionner une frequence..."
    )
    
    journal = st.selectbox(
        "Frequence de lecture de journaux",
        ["Pas du tout", "Parfois", "Souvent"],
        index=None,
        placeholder="Selectionner une frequence..."
    )

with col4:
    st.markdown('<p style="color: #8B4513; font-weight: 600; font-size: 1.05rem; margin-bottom: 0.8rem;">Composition familiale</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #FAF6F1; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <p style="margin: 0; color: #5C3D2E; font-size: 0.9rem;">
            Indiquez le nombre d enfants deja nes et vivants. 
            Cette valeur sera centree automatiquement par le modele.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    nb_vivants = st.slider(
        "Nombre d enfants vivants",
        min_value=0,
        max_value=15,
        value=0,
        step=1,
        help="Glissez pour selectionner le nombre exact"
    )

# Verification que tout est rempli
champs_remplis = all([
    region is not None,
    richesse is not None,
    proprietaire is not None,
    edu_femme is not None,
    edu_mari is not None,
    religion is not None,
    besoin_pf is not None,
    tv is not None,
    journal is not None
])

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Bouton de prediction
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    bouton_predire = st.button(
        "Obtenir la prediction",
        disabled=not champs_remplis,
        use_container_width=True
    )

if not champs_remplis:
    st.markdown("""
    <div style="text-align: center; color: #9E8E7E; font-size: 0.9rem; margin-top: 0.5rem;">
        Veuillez remplir tous les champs ci-dessus pour activer le bouton de prediction.
    </div>
    """, unsafe_allow_html=True)

# Prediction
if bouton_predire and champs_remplis:
    # Preparation des donnees
    donnees = pd.DataFrame({
        "edu_cat": [edu_femme],
        "richesse_cat": [richesse],
        "region_cat": [region],
        "besoin_pf_ns": [besoin_pf],
        "edu_mari_cat": [edu_mari],
        "religion_cat": [religion],
        "proprietaire_terre_cat": [proprietaire],
        "regarde_tv_cat": [tv],
        "regarde_journal_cat": [journal],
        "nb_vivants_centre": [nb_vivants]
    })
    
    prediction = modele.predict(donnees)[0]
    prediction_arrondie = round(prediction, 2)
    
    # Affichage du resultat
    st.markdown(f"""
    <div class="result-card">
        <div style="text-align: center;">
            <p style="font-family: 'Source Sans Pro', sans-serif; color: #6B5B4F; font-size: 1.1rem; margin-bottom: 0.5rem;">
                Nombre ideal d enfants predit
            </p>
            <div class="result-number">{prediction_arrondie}</div>
            <div class="result-label">enfants</div>
        </div>
        <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #E8DDD0;">
            <p style="color: #6B5B4F; font-size: 0.9rem; margin-bottom: 0.8rem;">
                <strong>Profil analyse :</strong>
            </p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.85rem; color: #5C3D2E;">
                <div>Region : <strong>{region}</strong></div>
                <div>Richesse : <strong>{richesse}</div>
                <div>Education femme : <strong>{edu_femme}</div>
                <div>Education conjoint : <strong>{edu_mari}</div>
                <div>Religion : <strong>{religion}</div>
                <div>Enfants vivants : <strong>{nb_vivants}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Interpretation contextuelle
    if prediction_arrondie <= 3:
        message = "Ce profil correspond a une fecondite desiree faible, typique des milieux urbains et eduques."
        couleur_msg = "#2E7D4F"
    elif prediction_arrondie <= 5:
        message = "Ce profil correspond a une fecondite desiree moderée, observee dans les milieux intermediaires."
        couleur_msg = "#B8860B"
    else:
        message = "Ce profil correspond a une fecondite desiree elevee, plus frequente dans les milieux ruraux et traditionnels."
        couleur_msg = "#8B4513"
    
    st.markdown(f"""
    <div style="background: #FAF6F1; border-radius: 12px; padding: 1.5rem; margin-top: 1.5rem; border-left: 4px solid {couleur_msg};">
        <p style="margin: 0; color: #4A2C17; font-size: 0.95rem; line-height: 1.6;">
            {message}
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Outil developpe dans le cadre d une etude demographique sur la fecondite au Cameroun</p>
    <p>Modele : Ridge Regression | Donnees : DHS Cameroun 2018</p>
</div>
""", unsafe_allow_html=True)
