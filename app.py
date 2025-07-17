from ai_model import predict_crop_insights  # <--- don't forget to import at top
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import random
import joblib
from forecast_utils import forecast_crop_prices
import os
import requests
from datetime import datetime, timezone, timedelta
import pickle
import numpy as np
#libraries concerning with locking the ai sections
from functools import wraps
from flask import redirect, url_for, session
from datetime import date

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Use a secure key in production

def subscription_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get("user_id")
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT end_date, is_active FROM subscriptions WHERE user_id = %s ORDER BY id DESC LIMIT 1
        """, (user_id,))
        row = cur.fetchone()
        cur.close()

        if row and row["is_active"] and date.today() <= row["end_date"]:
            return f(*args, **kwargs)
        return redirect(url_for("subscribe_page"))  # Redirect if not active

    return decorated_function  # ✅ This must be at the right level (not nested)

# ✅ This route must be outside (not indented into the function above)
@app.route('/subscribe_page')
def subscribe_page():
    return render_template('subscribe_page.html')

@app.route('/subscribe')
def subscribe():
    return render_template('subscribe.html') 


@app.route('/subscribe', methods=['POST'])
def process_subscription():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    plan = request.form.get('plan')
    user_id = session['user_id']

    # Determine subscription duration
    if plan == 'monthly':
        start_date = datetime.today()
        end_date = start_date + timedelta(days=30)
    elif plan == 'annual':
        start_date = datetime.today()
        end_date = start_date + timedelta(days=365)
    else:
        flash("Invalid plan selected.")
        return redirect(url_for('subscribe_page'))

    # Insert subscription into database
    cur = mysql.connection.cursor()
    cur.execute("""
        INSERT INTO subscriptions (user_id, start_date, end_date, trial, is_active)
        VALUES (%s, %s, %s, 0, 1)
    """, (user_id, start_date, end_date))
    mysql.connection.commit()
    cur.close()

    # Set session access
    session['subscription_active'] = True

    flash("✅ Subscription activated! You can now access premium features.")
    return redirect(url_for('dashboard'))


#log_activity
def log_activity(user_id, action):
    from datetime import datetime
    conn = mysql.connection
    cur = conn.cursor()
    cur.execute("INSERT INTO user_logs (user_id, action, timestamp) VALUES (%s, %s, %s)",
                (user_id, action, datetime.now()))
    conn.commit()
    cur.close()


# Load model and encoders
model = joblib.load('crop_multi_model.pkl')
mlb = joblib.load('crop_label_binarizer.pkl')
region_encoder = joblib.load('region_encoder.pkl')
district_encoder = joblib.load('district_encoder.pkl')
acidity_encoder = joblib.load('acidity_encoder.pkl')

# Utility: determine acidity level
def get_acidity_level(ph):
    if ph < 5.5:
        return "Highly Acidic"
    elif 5.5 <= ph <= 6.5:
        return "Moderately Acidic"
    elif 6.6 <= ph <= 7.3:
        return "Neutral"
    elif 7.4 <= ph <= 8.4:
        return "Slightly Alkaline"
    else:
        return "Highly Alkaline"

# Utility: get temperature and season (static or API)
def get_weather_info():
    temperature = 28  # Example static value
    month = datetime.now().month
    if 3 <= month <= 5:
        season = "Masika (Long Rainy)"
    elif 10 <= month <= 12:
        season = "Vuli (Short Rainy)"
    elif 6 <= month <= 9:
        season = "Dry Season"
    else:
        season = "Hot Season"
    return temperature, season

# Main route
@app.route("/recommendation", methods=["GET", "POST"])
def recommendation():
    if request.method == "POST":
        region = request.form["region"].strip()
        district = request.form["district"].strip()
        ph = float(request.form["ph"])

        try:
            acidity = get_acidity_level(ph)
            temperature, season = get_weather_info()

            region_enc = region_encoder.transform([region])[0]
            district_enc = district_encoder.transform([district])[0]
            acidity_enc = acidity_encoder.transform([acidity])[0]

            # Predict
            X = np.array([[region_enc, district_enc, ph, ph, acidity_enc]])
            prediction = model.predict(X)[0]
            crops = mlb.inverse_transform(np.array([prediction]))[0]

            best_crop = crops[0] if crops else "None"

            return render_template("recommendation.html", result={
                "region": region,
                "district": district,
                "ph": ph,
                "acidity": acidity,
                "temperature": temperature,
                "season": season,
                "recommended_crops": crops,
                "ml_prediction": best_crop
            })

        except Exception as e:
            return render_template("recommendation.html", error=f"⚠️ {str(e)}")

    return render_template("recommendation.html")



crop_data = {
    "Vitunguu": {
        "good_harvest": "Feb-Jul",
        "bad_harvest": "Jul-Nov",
        "good_planting": "Sept-Dec"
    },
    "Nyanya": {
        "good_harvest": "Jan-May",
        "bad_harvest": "Dec-Jun",
        "good_planting": "Sept-Dec"
    },
    "Hohokijani": {
        "good_harvest": "Feb-Apr",
        "bad_harvest": "Jun-Jan",
        "good_planting": "Oct-Nov"
    },
    "Karoti": {
        "good_harvest": "Oct-Mar",
        "bad_harvest": "Sep-Apr",
        "good_planting": "Jul-Oct"
    },
    "Matango": {
        "good_harvest": "Feb-May",
        "bad_harvest": "May-Jan",
        "good_planting": "Dec-Jan"
    },
    "Viazimviringo": {
        "good_harvest": "Mar-Jun",
        "bad_harvest": "Jul-Jan",
        "good_planting": "Dec-Feb"
    },
    "Tikitimaji": {
        "good_harvest": "Mar-Apr, Oct-Dec",
        "bad_harvest": "May-Sep, Dec-Feb",
        "good_planting": "Jan-Feb, Aug-Sept"
    },
    "Hohozarangi": {
        "good_harvest": "Jun-Nov",
        "bad_harvest": "May-Dec",
        "good_planting": "Feb-Mar"
    },
    "Tangawizi": {
        "good_harvest": "Apr-Jul",
        "bad_harvest": "Mar-Aug",
        "good_planting": "Dec-Feb"
    }
}


#MySQL Configuration
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = ""
app.config["MYSQL_DB"] = "signup"

mysql = MySQL(app)

@app.route('/')
@app.route('/index')
def index():
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    return render_template('index.html')

@app.route("/contact")
def contact():
    return render_template("contact.html")  # Make sure this file is inside the templates folder

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        # Check for admin login
        if username == 'admin' and password == 'admin123':
            session['admin'] = True
            flash("✅ Admin login successful!")
            return redirect(url_for('admin_dashboard'))

        # Otherwise, check normal user
        cur = mysql.connection.cursor()
        cur.execute("SELECT username, password FROM form WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()

        if user:
            stored_username, stored_password = user
            if password == stored_password:  # In production, use password hashing
                session['username'] = stored_username
                flash("✅ Login successful!")
                return redirect(url_for('dashboard'))
            else:
                flash("❌ Wrong password.")
        else:
            flash("❌ Username not found.")

    return render_template('login.html')



@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form['fullname'].strip()
        email = request.form['email'].strip()
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM form WHERE email = %s", (email,))
        existing_user = cur.fetchone()

        if existing_user:
            flash("❗ Email already exists. Try logging in.")
            return render_template('signup.html')

        # Insert new user
        cur.execute("""
            INSERT INTO form (fullname, email, username, password)
            VALUES (%s, %s, %s, %s)
        """, (fullname, email, username, password))
        mysql.connection.commit()

        # Get the new user's ID
        cur.execute("SELECT id FROM form WHERE email = %s", (email,))
        new_user_id = cur.fetchone()[0]

        # Insert free trial subscription
        trial_duration = timedelta(days=7)
        cur.execute("""
            INSERT INTO subscriptions (user_id, start_date, end_date, trial, is_active)
            VALUES (%s, %s, %s, 1, 1)
        """, (new_user_id, datetime.today(), datetime.today() + trial_duration))
        mysql.connection.commit()
        cur.close()

        flash("✅ Signup successful. Please log in.")
        return redirect(url_for('login'))

    return render_template('signup.html')



@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

    user_id = session.get('user_id')

    # Check if user has an active subscription
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT is_active, end_date FROM subscriptions
        WHERE user_id = %s ORDER BY end_date DESC LIMIT 1
    """, (user_id,))
    sub = cur.fetchone()
    cur.close()

    if sub and sub['is_active'] and sub['end_date'] >= date.today():
        session['subscription_active'] = True
    else:
        session['subscription_active'] = False

    return render_template('dashboard.html', username=session['username'])


@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin'):
        return redirect(url_for('login'))  # Secure check

    cur = mysql.connection.cursor()
    cur.execute("SELECT id, fullname, email, username FROM form")
    users = cur.fetchall()

    cur.execute("SELECT * FROM user_logs ORDER BY timestamp DESC LIMIT 50")
    logs = cur.fetchall()
    cur.close()

    return render_template('admin_dashboard.html', users=users, logs=logs)


@app.route('/market-trends', methods=['GET', 'POST'])
def market_trends():
    forecast = None
    crop = region = ""
    
    # These match  uploaded data files
    crops = [
        "maize", "rice", "beans", "sorghum", "bulrush_millet", "finger_millet", "round_potato"
    ]
    regions = [
        "dodoma", "arusha", "dar_es_salaam", "lindi", "morogoro", "iringa", "ruvuma", "tabora",
        "rukwa", "kigoma", "shinyanga", "mwanza", "kagera", "mara", "manyara", "njombe",
        "kilimanjaro", "singida", "geita", "songwe", "simiyu", "pwani"
    ]
    
    if request.method == 'POST':
        crop = request.form['crop']
        region = request.form['region']
        csv_file = f"full_market_trends_data/{crop}_{region}.csv"

        if os.path.exists(csv_file):
            forecast = forecast_crop_prices(csv_file)
            forecast = forecast.to_dict(orient="records")
        else:
            forecast = "not_found"
    
    return render_template("market_trends.html", forecast=forecast, crop=crop, region=region, crops=crops, regions=regions)



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Load dataset for crops
    df = pd.read_csv('planting_harvesting_times.csv')
    crops = sorted(df['Crop'].unique())

    if request.method == 'POST':
        selected_crop = request.form['crop']
        print(f"User selected crop: {selected_crop}")

        # Step 1: Smartly choose which forecast function to call based on the crop
        if selected_crop in ["cashew", "coffee", "sesame", "cotton", "maize", "rice", "sorghum", "millet", "banana", "avocado", "sunflower", "beans", "sugarcane", "tea", "apple"]:
            # Use generate_forecast for crops with predefined regions, harvest, and sell data
            forecast = generate_forecast(selected_crop)
        else:
            # Use predict_planting_harvest for crops in the crop_data dictionary
            forecast = predict_planting_harvest(selected_crop)

        return render_template('result.html', crop=selected_crop, forecast=forecast)

    return render_template('predict.html', crops=crops)

@app.route('/price-trends')
def price_trends():
    # You can later load from database or file, but for now we will hardcode the extracted data
    # (I'll help you structure this)
    return render_template('price_trends.html')



@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("🔓 You have been logged out.")
    return redirect(url_for('login'))
    


def predict_planting_harvest(crop_name):
    crop_data = {
        "Vitunguu": {
            "good_harvest": "Feb-Jul",
            "bad_harvest": "Jul-Nov",
            "good_planting": "Sept-Dec"
        },
        "Nyanya": {
            "good_harvest": "Jan-May",
            "bad_harvest": "Dec-Jun",
            "good_planting": "Sept-Dec"
        },
        "Hohokijani": {
            "good_harvest": "Feb-April",
            "bad_harvest": "Jun-Jan",
            "good_planting": "Oct-Nov"
        },
        "Karoti": {
            "good_harvest": "Oct-Mar",
            "bad_harvest": "Sep-Apr",
            "good_planting": "Jul-Oct"
        },
        "Matango": {
            "good_harvest": "Feb-May",
            "bad_harvest": "May-Jan",
            "good_planting": "Dec-Jan"
        },
        "Viazimviringo": {
            "good_harvest": "Mar-Jun",
            "bad_harvest": "Jul-Jan",
            "good_planting": "Dec-Feb"
        },
        "Tikitimaji": {
            "good_harvest": "Mar-April, Okt-Dec",
            "bad_harvest": "Mei-Sep, Des-Feb",
            "good_planting": "Jan-Feb, Aug-Sept"
        },
        "Hohozarangi": {
            "good_harvest": "Jun-Nov",
            "bad_harvest": "May-Dec",
            "good_planting": "Feb-Mar"
        },
        "Tangawizi": {
            "good_harvest": "Apr-Jul",
            "bad_harvest": "Mar-Aug",
            "good_planting": "Dec-Feb"
        }
    }
    
    if crop_name not in crop_data:
        return {
            "good_harvest": "N/A",
            "bad_harvest": "N/A",
            "good_planting": "N/A"
        }
    
    return crop_data[crop_name]


if __name__ == '__main__':
    app.run(debug=True)   