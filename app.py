from flask import Flask, render_template, request, jsonify
import pandas as pd
import google.generativeai as genai
import re

# 🔹 Configure API Key for Google Gemini AI
API_KEY = "AIzaSyB2u8nvt8SxnHxCxBk8oqDKlAZ97sv-5mw"  # Replace with your Gemini API Key
genai.configure(api_key=API_KEY)

# 🔹 Load CSV dataset with bug fixes
DATASET_FILE = r"C:\Users\MADHAV BHALODIA\Vscode(py)\hackathon\soil_predictions_full.csv"

try:
    # Load CSV dataset
    df = pd.read_csv(DATASET_FILE)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['records'] = df['records'].str.strip().str.lower()

    print("✅ Dataset loaded successfully!")

except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()


# 🔹 Function to get soil parameters
def get_soil_parameters(record):
    """Retrieve soil parameters from the CSV dataset."""
    record = record.strip().lower()

    # Perform case-insensitive and partial matching
    soil_row = df[df['records'].str.contains(record, case=False, na=False)]

    if soil_row.empty:
        return None

    return soil_row.iloc[0].to_dict()


# 🔹 Function to format text with rich styling
def format_rich_text(text):
    """Convert Markdown-like syntax to HTML for rich text rendering."""
    # **bold** → <strong>bold</strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

    # *italic* → <em>italic</em>
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)

    # Bullet points: - item → <li>item</li>
    text = re.sub(r'- (.*?)\n', r'<li>\1</li>', text)
    text = re.sub(r'(<li>.*?</li>)', r'<ul>\1</ul>', text)

    # Newlines → <br>
    text = text.replace('\n', '<br>')

    return text


# 🔹 Function to analyze soil and advise using Gemini AI
def analyze_soil_and_advise(record):
    """Perform soil analysis using Gemini AI."""
    soil_params = get_soil_parameters(record)

    if not soil_params:
        return f"⚠️ No records found for '{record}'. Please check the input."

    # Construct the prompt with soil parameters
    prompt = f"""
    Based on the soil type '{record}', here are the recorded values:

    - **Capacity Moisture:** {soil_params.get('predicted_capacitity_moist', 'N/A')}%
    - **Temperature:** {soil_params.get('predicted_temp', 'N/A')}°C
    - **Moisture Level:** {soil_params.get('predicted_moist', 'N/A')}%
    - **Electrical Conductivity (EC):** {soil_params.get('predicted_ec_(u/10_gram)', 'N/A')} u/10g
    - **pH Level:** {soil_params.get('predicted_ph', 'N/A')}
    - **Nitrogen Content:** {soil_params.get('predicted_nitro_(mg/10_g)', 'N/A')} mg/10g
    - **Phosphorus Content:** {soil_params.get('predicted_posh_nitro_(mg/10_g)', 'N/A')} mg/10g
    - **Potassium Content:** {soil_params.get('predicted_pota_nitro_(mg/10_g)', 'N/A')} mg/10g

    🌿 **Soil Analysis & Recommendations:**
    - Assess soil health based on pH, EC, and nutrient levels.
    - Determine crop suitability.
    - Suggest necessary improvements (e.g., adding compost, adjusting pH).
    - Provide irrigation guidance.
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)

        if response and hasattr(response, 'text'):
            rich_text = format_rich_text(response.text)
            return rich_text
        else:
            return "⚠️ Gemini AI did not return a valid response. Try again."

    except Exception as e:
        return f"⚠️ Gemini AI Error: {e}"


# 🔹 Flask Routes
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle soil analysis requests."""
    record = request.form.get('record')

    if not record:
        return jsonify({'error': 'Please provide a record name.'})

    # Perform soil analysis
    analysis = analyze_soil_and_advise(record)

    return jsonify({'result': analysis})


if __name__ == '__main__':
    app.run(debug=True)
