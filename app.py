from flask import Flask, render_template, request, redirect, url_for, session
import os
import pandas as pd
import joblib
import numpy as np
from integmodel import load_model, predict_image
from prediction_code import predict  # Growth analysis function
from pest_detection import detect_pests  # Pest detection function
from weed_classifier import load_model_and_mapping, predict_weed  # Weed classification functions
from yield_prediction import predict_yield, predict_rice_yield
from weatheralerts import IndianLocations, WeatherMonitor

app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'

locations = IndianLocations()
weather_monitor = WeatherMonitor(locations)

# Simulate a user database (replace with a real database in production)
users = {}  # Format: email: password

@app.route('/', methods=['GET', 'POST'])
def register_and_login():
    """Single page for registration and login."""
    if request.method == 'POST':
        # Fetch form data
        name = request.form.get('name')
        mobile = request.form.get('mobile')
        email = request.form['email']
        password = request.form['password']
        city = request.form.get('city')

        # Handle registration
        if name and mobile and city:
            if email not in users:
                users[email] = {'password': password, 'name': name, 'mobile': mobile, 'city': city}
                session['user'] = email
                session['location'] = city
                return redirect(url_for('index'))
            return "User already exists. Please log in!"

        # Handle login
        if email in users and users[email]['password'] == password:
            session['user'] = email
            session['location'] = users[email]['city']
            return redirect(url_for('index'))
        return "Invalid credentials or registration required!"

    # Fetch cities from dataset
    cities = [(city.title(), info['state']) for city, info in locations.cities.items()]
    return render_template('register_and_login.html', cities=cities)

RESULTS_TEXT_FILE = 'static/analysis_results.txt'

def read_results_from_text_file():
    """Read analysis results from the text file."""
    if not os.path.exists(RESULTS_TEXT_FILE):
        return []

    results = []
    with open(RESULTS_TEXT_FILE, 'r') as file:
        current_result = {}
        for line in file:
            line = line.strip()
            if line.startswith("Analysis"):
                if current_result:
                    results.append(current_result)
                current_result = {"ID": len(results) + 1}
            elif line.startswith("Growth Health Analysis:"):
                current_result["Growth Health Analysis"] = line.split(":", 1)[1].strip()
            elif line.startswith("Pest Detection:"):
                current_result["Pest Detection"] = line.split(":", 1)[1].strip()
            elif line.startswith("Weed Severity Classification:"):
                current_result["Weed Severity Classification"] = line.split(":", 1)[1].strip()
        if current_result:
            results.append(current_result)
    return results

@app.route('/dashboard')
def dashboard():
    """Display saved analysis results in a table format."""
    analysis_results = read_results_from_text_file()
    return render_template('dashboard.html', analysis_results=analysis_results)




@app.route('/index')
def index():
    """Main index page displaying weather and analysis options."""
    if 'user' not in session or 'location' not in session:
        return redirect(url_for('register_and_login'))
    
    location = session['location']
    city_info = locations.search_location(location.lower())[1]

    # Fetch weather data
    weather_data = weather_monitor.get_weather_data(*city_info['coords'])
    if not weather_data or 'current' not in weather_data:
        weather_data = {
            'current': {
                'temperature_2m': 'N/A',
                'relative_humidity_2m': 'N/A',
                'wind_speed_10m': 'N/A'
            }
        }
        alerts = ["Weather data unavailable."]
        recommendations = ["Please check back later for recommendations."]
    else:
        # Generate alerts and recommendations
        alerts = weather_monitor.generate_alerts(weather_data)
        recommendations = weather_monitor.generate_recommendations(weather_data)

    return render_template(
        'index.html',
        location=location,
        weather=weather_data,
        alerts=alerts,
        recommendations=recommendations,
    )

@app.route('/logout')
def logout():
    """Logout route."""
    session.clear()
    return redirect(url_for('register_and_login'))

# Threshold values


# Load Weed Severity Classification Model
WEED_MODEL_PATH = r'C:\sreejha_project\project\models\best_model.pth'
WEED_SPLIT_INFO_PATH = r'C:\sreejha_project\project\models\split_info.json'

# Load encoders and models
state_encoder = joblib.load('models/state_encoder.pkl')
district_encoder = joblib.load('models/district_encoder.pkl')
season_encoder = joblib.load('models/season_encoder.pkl')

# Ensure the model and mappings are loaded correctly
try:
    weed_model, weed_idx_to_label, weed_device = load_model_and_mapping(WEED_MODEL_PATH, WEED_SPLIT_INFO_PATH)
except Exception as e:
    print(f"Error loading weed model: {e}")
    weed_model, weed_idx_to_label, weed_device = None, None, None


model = load_model(r'C:\sreejha_project\project\models\best_model_checkpoint.pth')
def analyze_image(image_path):
    """Run integmodel.py first to determine which analysis to perform."""
    class_names = ['growth', 'pest', 'weed']
    predicted_class = predict_image(model, image_path)
    predicted_label = class_names[predicted_class]
    
    print(f"[INFO] Integmodel Prediction: {predicted_label}")
    return predicted_label


@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image = request.files['image']
        image_path = os.path.join('static/uploads', image.filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image.save(image_path)

        # Determine which model to run
        predicted_label = analyze_image(image_path)

        # Default "No results" for all
        growth_results = {"Predicted Label": "No results", "Confidence": "No results"}
        pest_results = {"No pests detected": ["No results"]}
        weed_results = {"top_prediction": "No results", "all_predictions": []}

        # Run only the relevant model
        if predicted_label == "growth":
            predicted_label, growth_confidence = predict(image_path)
            growth_results = {"Predicted Label": predicted_label, "Confidence": f"{growth_confidence:.2f}%"}
        elif predicted_label == "pest":
            pest_results = detect_pests(image_path) or {"No pests detected": ["No results"]}
        elif predicted_label == "weed":
            weed_results = predict_weed(image_path, weed_model, weed_idx_to_label, weed_device)
            weed_results = {"top_prediction": weed_results["top_prediction"], "all_predictions": weed_results["all_predictions"]}
            
        

        print("--- Final Analysis Results ---")
        print(f"Growth: {growth_results}")
        print(f"Pest: {pest_results}")
        print(f"Weed: {weed_results}")
        
        
        # Generate Recommendations
        fertilizer_recommendation = None
        herbicide_recommendation = None

        if growth_results["Predicted Label"] == "Nutrient-Deficient":
            fertilizer_recommendation = "Apply NPK fertilizer with a ratio of 10:26:26."
        elif growth_results["Predicted Label"] == "Healthy":
            fertilizer_recommendation = "No additional fertilizer is needed at this stage."
        elif growth_results["Predicted Label"] == "Pest-Affected":
            fertilizer_recommendation = "Use balanced fertilizer to strengthen plant immunity."

        if weed_results["top_prediction"] != "No results":
            if weed_results["top_prediction"]["class"] == "High":
                herbicide_recommendation = "Apply a pre-emergent herbicide like Pendimethalin."
            elif weed_results["top_prediction"]["class"] == "Medium":
                herbicide_recommendation = "Use a post-emergent herbicide such as Glyphosate."
            elif weed_results["top_prediction"]["class"] == "Low":
                herbicide_recommendation = "Manual weeding is sufficient."
        
        
        # Save results to text file
        with open(RESULTS_TEXT_FILE, 'a') as file:
            file.write(f"Analysis {len(read_results_from_text_file()) + 1}:\n")
            file.write(f"Growth Health Analysis: {growth_results}\n")
            file.write(f"Pest Detection: {pest_results}\n")
            file.write(f"Weed Severity Classification: {weed_results}\n")
            file.write("-" * 50 + "\n")
            
            

        return render_template('analysis_results.html', image_path=image_path,
                               growth_results=growth_results, pest_results=pest_results,
                               weed_results=weed_results,
                               fertilizer_recommendation=fertilizer_recommendation,
                               herbicide_recommendation=herbicide_recommendation)
        
    

    return render_template('upload_image.html')





@app.route('/yield_prediction', methods=['GET', 'POST'])
def yield_prediction():
    return render_template('yield_prediction.html')



@app.route('//yield_prediction_Production', methods=['GET', 'POST'])
def yield_prediction_state():
    # Get available options from encoders
    states = state_encoder.classes_.tolist()
    districts = district_encoder.classes_.tolist()
    seasons = season_encoder.classes_.tolist()

    result = None
    if request.method == 'POST':
        try:
            # Collect user input
            user_input = {
                'State': request.form.get('state', '').strip(),
                'District': request.form.get('district', '').strip(),
                'Year': int(request.form.get('year', 0)),
                'Season': request.form.get('season', '').strip(),
                'Area': float(request.form.get('area', 0)),
                'Production': float(request.form.get('production', 0))
            }

            # Predict yield
            predicted_yield, confidence = predict_yield(user_input)

            if predicted_yield is None:
                result = {"error": "Invalid input or prediction failed. Please try again."}
            else:
                result = {
                    "predicted_yield": f"{predicted_yield:.2f} Tonne/Hectare",
                    "confidence_interval": f"[{confidence[0]:.2f}, {confidence[1]:.2f}] Tonne/Hectare"
                }
        except Exception as e:
            result = {"error": f"An error occurred: {e}"}

    return render_template(
        'yield_prediction_state.html',
        states=states,
        districts=districts,
        seasons=seasons,
        result=result
    )

@app.route('/yield_prediction_Fertilizer', methods=['GET', 'POST'])
def yield_prediction_fertilizer():
    result = None
    if request.method == 'POST':
        try:
            predicted_yield = predict_rice_yield(
                float(request.form.get('nitrogen', 0)),
                float(request.form.get('phosphorus', 0)),
                float(request.form.get('potassium', 0))
            )
            result = {"predicted_yield": f"{predicted_yield:.2f} Tonne/Hectare"}
        except Exception as e:
            result = {"error": f"An error occurred: {e}"}

    return render_template('yield_prediction_fertilizer.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
