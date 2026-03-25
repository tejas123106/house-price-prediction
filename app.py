from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("bangalore_house_price_model.pkl", "rb"))

# All features
feature_names = [
    'number of bedroom', 'number of bathroom', 'age', 'sqrt_feet', 'parking',
    'location_Arekere', 'location_BTM Layout', 'location_Banashankari',
    'location_Bannerghatta Road', 'location_Basavanagudi', 'location_Bellandur',
    'location_Bidadi', 'location_Bommanahalli', 'location_CV Raman Nagar',
    'location_Chikkajala', 'location_Cunningham Road', 'location_Devanahalli',
    'location_Doddaballapur', 'location_Domlur', 'location_Electronic City',
    'location_Frazer Town', 'location_HSR Layout', 'location_Hebbal',
    'location_Horamavu', 'location_Hosakote', 'location_Indiranagar',
    'location_JP Nagar', 'location_Jayanagar', 'location_KR Puram',
    'location_Kanakapura Road', 'location_Kengeri', 'location_Koramangala',
    'location_Kundalahalli', 'location_Magadi Road', 'location_Malleshwaram',
    'location_Marathahalli', 'location_Mysore Road', 'location_Nagarbhavi',
    'location_Nelamangala', 'location_New BEL Road',
    'location_Old Airport Road', 'location_Panathur', 'location_RT Nagar',
    'location_Rajajinagar', 'location_Ramamurthy Nagar',
    'location_Richmond Town', 'location_Sahakar Nagar',
    'location_Sarjapur Road', 'location_Shivajinagar', 'location_Thubarahalli',
    'location_Uttarahalli', 'location_Varthur', 'location_Vijayanagar',
    'location_Whitefield', 'location_Yelahanka'
]

# Dropdown list
locations = [x.replace("location_", "") for x in feature_names if x.startswith("location_")]

@app.route("/")
def home():
    return render_template("index.html", locations=locations)


@app.route("/predict", methods=["POST"])
def predict():

    print("FORM RECEIVED:", request.form)  # Debug print

    try:
        bedrooms = request.form.get("bedrooms")
        bathrooms = request.form.get("bathrooms")
        age = request.form.get("age")
        sqft = request.form.get("sqft")
        parking = request.form.get("parking")
        location = request.form.get("location")

        print("Values:", bedrooms, bathrooms, age, sqft, parking, location)

        if None in [bedrooms, bathrooms, age, sqft, parking, location]:
            return "❌ Flask did NOT receive some fields. Check terminal output."

        bedrooms = float(bedrooms)
        bathrooms = float(bathrooms)
        age = float(age)
        sqft = float(sqft)
        parking = float(parking)

    except Exception as e:
        return f"❌ Error: {e}"

    # Create input vector
    input_data = np.zeros(len(feature_names))

    # Assign numeric values
    input_data[feature_names.index("number of bedroom")] = bedrooms
    input_data[feature_names.index("number of bathroom")] = bathrooms
    input_data[feature_names.index("age")] = age
    input_data[feature_names.index("sqrt_feet")] = sqft
    input_data[feature_names.index("parking")] = parking

    # One-hot encode location
    loc_feature = "location_" + location
    if loc_feature in feature_names:
        input_data[feature_names.index(loc_feature)] = 1

    price = model.predict([input_data])[0]

    return render_template("result.html", price=round(price, 2))


if __name__ == "__main__":
    app.run(debug=True)
