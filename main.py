import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from st_files_connection import FilesConnection
import json

# Feature groups
HORSE_FEATURES = st.secrets["HORSE_FEATURES"]
JOCKEY_FEATURES = st.secrets["JOCKEY_FEATURES"]
TRAINER_FEATURES = st.secrets["TRAINER_FEATURES"]

@st.cache_data(ttl=2400)
def load_model(model_path):
    model = XGBClassifier()
    conn = st.connection('s3', type=FilesConnection)
    file_content = conn.read(model_path, input_format="text", ttl=600)
    
    try:
        # Parse the JSON content
        model_dict = json.loads(file_content)
        # Save the parsed JSON to a temporary file
        with open('temp_model.json', 'w') as f:
            json.dump(model_dict, f)
        # Load the model from the temporary file
        model.load_model('temp_model.json')
        # Remove the temporary file
        import os
        os.remove('temp_model.json')
        return model
    
    except json.JSONDecodeError:
        st.error("The file content is not valid JSON. Please check the file format.")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")
    return None

@st.cache_data(ttl=600)
def load_data(data_path):
    conn = st.connection('s3', type=FilesConnection)
    df = conn.read(data_path, input_format="csv", ttl=600)
    return df


def prepare_features(features: pd.DataFrame, model: XGBClassifier) -> pd.DataFrame:
    required_columns = model.get_booster().feature_names
    for col in required_columns:
        if col not in features.columns:
            features[col] = 0
    return features[required_columns]


def predict(model: XGBClassifier, features: pd.DataFrame, importance_factors: dict) -> pd.DataFrame:
    # Apply importance factors
    for feature_group, factor in importance_factors.items():
        for feature in globals()[f"{feature_group.upper()}_FEATURES"]:
            if feature in features.columns:
                features[feature] *= factor

    predictions = model.predict_proba(features)
    return pd.DataFrame(predictions, columns=model.classes_)


# Streamlit UI
st.title("Horse Race Prediction with Adjustable Feature Importance")
# Load model and data
model = load_model('dg-horse-data/fr/models/classification_model.json')
data = load_data('dg-horse-data/fr/models/dataset.csv')
selected = st.multiselect("Select race name", data['race_name'].unique())
# Feature importance sliders
st.sidebar.header("Adjust Feature Importance")
horse_importance = st.sidebar.slider("Horse Importance", 0.5, 2.0, 1.0, 0.1)
jockey_importance = st.sidebar.slider("Jockey Importance", 0.5, 2.0, 1.0, 0.1)
trainer_importance = st.sidebar.slider("Trainer Importance", 0.5, 2.0, 1.0, 0.1)

importance_factors = {
    "horse": horse_importance,
    "jockey": jockey_importance,
    "trainer": trainer_importance
}

# Prepare features and make predictions
features = prepare_features(data, model)
predictions = predict(model, features, importance_factors)
predictions.index = data.index
data['probability'] = predictions.iloc[:, 0]
# Normalize probabilities within each race to ensure they sum to 1
data['prob_final'] = data['probability'] / data.groupby('race_id')['probability'].transform('sum')
data['prediction (win chance in %)'] = (data['prob_final'] * 100).round(2)

# Style the DataFrame for display with percentage sign
styled_data = data.style.format({'prediction (win chance in %)': '{:.2f}%'})
race_df = data[['race_name', 'city', 'date', 'horse', 'jockey', 'trainer', 'prediction (win chance in %)']]
race_df['date'] = pd.to_datetime(race_df['date'])
if selected:
    for race in selected:
        data_display = race_df[race_df['race_name'] == race]
        race_date = data_display['date'].iloc[0].strftime('%Y-%m-%d')
        city = data_display['city'].iloc[0]
        data_display = data_display.drop(columns=['city', 'date', 'race_name'])
        st.markdown(f"### {race}")
        st.markdown(f"**Date:** {race_date} | **City:** {city}")
        st.dataframe(data_display)
        st.markdown("---")
else:
    st.info("Please select at least one race name to display the data.")