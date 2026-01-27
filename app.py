import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load saved model & encoders
# -----------------------------
model = joblib.load("random_forest_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
model_features = joblib.load("model_features.pkl")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Forest Cover Type Predictor",
    layout="centered"
)

st.title("🌲 Forest Cover Type Prediction")
st.write(
    "Enter terrain and environmental features to predict the forest cover type."
)

st.divider()

# -----------------------------
# Input features
# -----------------------------
st.subheader("📥 Input Features")

col1, col2 = st.columns(2)

with col1:
    Elevation = st.number_input("Elevation", min_value=0, value=2800)
    Aspect = st.number_input("Aspect", min_value=0, max_value=360, value=45)
    Slope = st.number_input("Slope", min_value=0, value=12)
    Horizontal_Distance_To_Hydrology = st.number_input(
        "Horizontal Distance To Hydrology", min_value=0, value=120
    )
    Vertical_Distance_To_Hydrology = st.number_input(
        "Vertical Distance To Hydrology", value=10
    )

with col2:
    Horizontal_Distance_To_Roadways = st.number_input(
        "Horizontal Distance To Roadways", min_value=0, value=300
    )
    Hillshade_9am = st.number_input("Hillshade at 9 AM", min_value=0, max_value=255, value=210)
    Hillshade_Noon = st.number_input("Hillshade at Noon", min_value=0, max_value=255, value=230)
    Hillshade_3pm = st.number_input("Hillshade at 3 PM", min_value=0, max_value=255, value=180)
    Horizontal_Distance_To_Fire_Points = st.number_input(
        "Horizontal Distance To Fire Points", min_value=0, value=450
    )

# -----------------------------
# Create input DataFrame
# -----------------------------
X_new = pd.DataFrame([{
    "Elevation": Elevation,
    "Aspect": Aspect,
    "Slope": Slope,
    "Horizontal_Distance_To_Hydrology": Horizontal_Distance_To_Hydrology,
    "Vertical_Distance_To_Hydrology": Vertical_Distance_To_Hydrology,
    "Horizontal_Distance_To_Roadways": Horizontal_Distance_To_Roadways,
    "Hillshade_9am": Hillshade_9am,
    "Hillshade_Noon": Hillshade_Noon,
    "Hillshade_3pm": Hillshade_3pm,
    "Horizontal_Distance_To_Fire_Points": Horizontal_Distance_To_Fire_Points,
}])

# Align with training features
X_new_aligned = X_new.reindex(
    columns=model_features,
    fill_value=0
)

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Forest Cover Type"):
    # Predict class
    prediction_encoded = model.predict(X_new_aligned)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

    # Predict probabilities
    probabilities = model.predict_proba(X_new_aligned)[0]
    confidence = max(probabilities) * 100

    # Show main result
    st.success(f"🌲 **Predicted Forest Cover Type:** {prediction_label}")
    st.info(f"📊 **Prediction Confidence:** {confidence:.2f}%")

    # -----------------------------
    # Top 3 predictions
    # -----------------------------
    st.subheader("🔮 Top 3 Predictions")

    class_names = label_encoder.inverse_transform(
        range(len(probabilities))
    )

    top3 = sorted(
        zip(class_names, probabilities),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    for cls, prob in top3:
        st.write(f"• **{cls}** : {prob*100:.2f}%")
