# import pandas as pd
# import numpy as np
# from flask import Flask, request, jsonify
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import SelectKBest, mutual_info_classif
# from flask_cors import CORS

# import os

# from sklearn.ensemble import VotingClassifier, RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# import joblib



# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)



# # Load and preprocess data
# train_data = pd.read_csv("training_data.csv")
# test_data = pd.read_csv("test_data.csv")

# def clean_dataset(df):
#     df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
#     df.columns = df.columns.str.strip().str.lower()
#     return df

# train_data = clean_dataset(train_data)
# test_data = clean_dataset(test_data)

# X_train = train_data.drop('prognosis', axis=1)
# y_train = train_data['prognosis']
# X_test = test_data.drop('prognosis', axis=1)
# y_test = test_data['prognosis']

# # Feature selection
# selector = SelectKBest(score_func=mutual_info_classif, k=120)
# X_train_selected = selector.fit_transform(X_train, y_train)
# X_test_selected = selector.transform(X_test)
# selected_features = X_train.columns[selector.get_support()].tolist()

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_selected)
# X_test_scaled = scaler.transform(X_test_selected)

# # Initialize and train multiple models
# models = {
#     'svm': SVC(kernel='rbf', probability=True, random_state=42),
#     'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
#     'decision_tree': DecisionTreeClassifier(max_depth=5, random_state=42),
#     'naive_bayes': GaussianNB(),
#     'knn': KNeighborsClassifier(n_neighbors=5)
# }

# # Train individual models
# for name, model in models.items():
#     model.fit(X_train_scaled, y_train)
#     y_pred = model.predict(X_test_scaled)
#     print(f"{name} accuracy: {accuracy_score(y_test, y_pred)}")

# # Create ensemble model
# ensemble = VotingClassifier(
#     estimators=[(name, model) for name, model in models.items()],
#     voting='soft'
# )
# ensemble.fit(X_train_scaled, y_train)
# ensemble_accuracy = accuracy_score(y_test, ensemble.predict(X_test_scaled))
# print(f"Ensemble accuracy: {ensemble_accuracy}")



# # Specialist mapping
# disease_to_specialist = {
#     "Fungal infection": "Dermatologist",
#     "Allergy": "Allergist",
#     "GERD": "Gastroenterologist",
#     "Chronic cholestasis": "Gastroenterologist",
#     "Drug Reaction": "Dermatologist",
#     "Peptic ulcer disease": "Gastroenterologist",
#     "AIDS": "Infectious Disease Specialist",
#     "Diabetes": "Endocrinologist",
#     "Gastroenteritis": "Gastroenterologist",
#     "Bronchial Asthma": "Pulmonologist",
#     "Hypertension": "Cardiologist",
#     "Migraine": "Neurologist",
#     "Cervical spondylosis": "Orthopedics",
#     "Paralysis (brain hemorrhage)": "Neurologist",
#     "Jaundice": "Gastroenterologist",
#     "Malaria": "Infectious Disease Specialist",
#     "Chicken pox": "Medicine",
#     "Dengue": "Infectious Disease Specialist",
#     "Typhoid": "Infectious Disease Specialist",
#     "Hepatitis A": "Hepatologist",
#     "Hepatitis B": "Hepatologist",
#     "Hepatitis C": "Hepatologist",
#     "Hepatitis D": "Hepatologist",
#     "Hepatitis E": "Hepatologist",
#     "Alcoholic hepatitis": "Gastroenterologist",
#     "Tuberculosis": "Pulmonologist",
#     "Common Cold": "Medicine",
#     "Pneumonia": "Pulmonologist",
#     "Dimorphic hemorrhoids(piles)": "Proctologist",
#     "Heart attack": "Cardiologist",
#     "Varicose veins": "General_Surgeon",
#     "Hypothyroidism": "Endocrinologist",
#     "Hyperthyroidism": "Endocrinologist",
#     "Hypoglycemia": "Endocrinologist",
#     "Osteoarthritis": "Orthopedics",
#     "Arthritis": "Orthopedics",
#     "(vertigo) Paroxysmal Positional Vertigo": "Neurologist",
#     "Acne": "Dermatologist",
#     "Urinary tract infection": "Urologist",
#     "Psoriasis": "Dermatologist",
#     "Impetigo": "Dermatologist"
# }


    

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         input_symptoms = data.get("symptoms", [])

#         input_vector = {symptom: 0 for symptom in selected_features}
#         for symptom in input_symptoms:
#             if symptom in input_vector:
#                 input_vector[symptom] = 1

#         input_df = pd.DataFrame([input_vector])
#         input_scaled = scaler.transform(input_df)

#         # Use ensemble model for prediction
#         predicted_disease = ensemble.predict(input_scaled)[0]
#         probabilities = ensemble.predict_proba(input_scaled)[0]
        
#         specialist = disease_to_specialist.get(predicted_disease, "General Practitioner")

#         return jsonify({
#             "predicted_disease": predicted_disease,
#             "recommended_specialist": specialist,
#             "confidence": float(np.max(probabilities))
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # if __name__ == '__main__':
# #     app.run(port=8000, debug=True)




# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import psycopg2
# import bcrypt
# from datetime import datetime

# # app = Flask(__name__)
# # CORS(app)

# # PostgreSQL connection
# conn = psycopg2.connect(
    # host="localhost",
    # database="JNMC",
    # user="postgres",
    # password="Anas@1234",
    # port=5432
# )
# cur = conn.cursor()

# allowed_tables = ["Cardiologist", "Dermatologist", "General_surgeon", "Endocrinologist",
#                   "Gastroenterologist", "Medicine", "Haematology", "Neurologist", "Orthopedics", "Urologist"]

# @app.route("/")
# def home():
#     return "Flask Backend is running"

# # Signup
# @app.route("/api/signup", methods=["POST"])
# def signup():
#     data = request.get_json()
#     name = data["name"]
#     email = data["email"]
#     password = data["password"]

#     cur.execute("SELECT * FROM users WHERE email = %s", (email,))
#     if cur.fetchone():
#         return jsonify({"error": "Email already in use"}), 400

#     hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
#     cur.execute("INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s) RETURNING id, name, email",
#                 (name, email, hashed.decode()))
#     user = cur.fetchone()
#     conn.commit()
#     return jsonify({"message": "User registered successfully", "user": {
#         "id": user[0], "name": user[1], "email": user[2]
#     }})

# # Login
# @app.route("/api/login", methods=["POST"])
# def login():
#     data = request.get_json()
#     email = data["email"]
#     password = data["password"]

#     cur.execute("SELECT * FROM users WHERE email = %s", (email,))
#     user = cur.fetchone()

#     if not user or not bcrypt.checkpw(password.encode('utf-8'), user[3].encode()):
#         return jsonify({"error": "Invalid email or password"}), 401

#     return jsonify({"message": "Login successful", "user": {
#         "id": user[0], "name": user[1], "email": user[2]
#     }})

# # Get OPD doctor list and location
# @app.route("/api/opd/<opd_name>", methods=["GET"])
# def get_opd(opd_name):
#     if opd_name not in allowed_tables:
#         return jsonify({"error": "Invalid OPD name"}), 400

#     cur.execute(f"SELECT * FROM {opd_name}")
#     doctors = cur.fetchall()
#     doctor_columns = [desc[0] for desc in cur.description]  # <-- Move this up

#     cur.execute("SELECT opd_number, address FROM opd_location WHERE opd_name = %s", (opd_name,))
#     location = cur.fetchone()
#     opd_number = location[0] if location else "N/A"
#     address = location[1] if location else "N/A"

#     doctor_list = [dict(zip(doctor_columns, row)) for row in doctors]

#     for doc in doctor_list:
#         doc["opd_number"] = opd_number
#         doc["address"] = address

#     return jsonify(doctor_list)

# # Store prediction
# @app.route("/api/store-prediction", methods=["POST"])
# def store_prediction():
#     data = request.get_json()
#     required = ["user_id", "disease", "specialist"]
#     if not all(k in data for k in required):
#         return jsonify({"error": "Missing required fields"}), 400

#     cur.execute("""
#         INSERT INTO disease_prediction 
#         (user_id, symptom1, symptom2, symptom3, symptom4, symptom5, disease, specialist, predicted_at)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
#         RETURNING *;
#     """, (data["user_id"], data.get("symptom1"), data.get("symptom2"), data.get("symptom3"),
#           data.get("symptom4"), data.get("symptom5"), data["disease"], data["specialist"]))

#     conn.commit()
#     return jsonify({"message": "Prediction stored"})

# # User history
# @app.route("/api/user-history/<int:user_id>", methods=["GET"])
# def user_history(user_id):
#     cur.execute("""
#         SELECT symptom1, symptom2, symptom3, symptom4, symptom5, disease, specialist, predicted_at
#         FROM disease_prediction
#         WHERE user_id = %s
#         ORDER BY predicted_at DESC;
#     """, (user_id,))
#     rows = cur.fetchall()
#     if not rows:
#         return jsonify({"message": "No history found for this user."}), 404

#     return jsonify([
#         {
#             "symptom1": row[0], "symptom2": row[1], "symptom3": row[2],
#             "symptom4": row[3], "symptom5": row[4],
#             "disease": row[5], "specialist": row[6], "predicted_at": row[7].isoformat()
#         } for row in rows
#     ])

# # Admin: all user history
# @app.route("/api/all-history", methods=["GET"])
# def all_history():
#     cur.execute("""
#         SELECT 
#             dp.id, u.id, u.name, u.email,
#             dp.symptom1, dp.symptom2, dp.symptom3, dp.symptom4, dp.symptom5,
#             dp.disease, dp.specialist, dp.predicted_at
#         FROM disease_prediction dp
#         JOIN users u ON dp.user_id = u.id
#         ORDER BY dp.predicted_at DESC;
#     """)
#     rows = cur.fetchall()
#     if not rows:
#         return jsonify({"message": "No history records found."}), 404

#     return jsonify([
#         {
#             "prediction_id": r[0], "user_id": r[1], "user_name": r[2], "user_email": r[3],
#             "symptom1": r[4], "symptom2": r[5], "symptom3": r[6],
#             "symptom4": r[7], "symptom5": r[8],
#             "disease": r[9], "specialist": r[10], "predicted_at": r[11].isoformat()
#         } for r in rows
#     ])

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)




import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from flask_cors import CORS
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import psycopg2
from psycopg2 import pool
import bcrypt
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Database connection pool
try:
    postgresql_pool = psycopg2.pool.SimpleConnectionPool(
        1, 20,
        dsn=os.getenv('POSTGRES_URL')
    )
    print("Connection pool created successfully")
except Exception as e:
    print(f"Error while creating connection pool: {e}")
    postgresql_pool = None

# Load and preprocess data
train_data = pd.read_csv("training_data.csv")
test_data = pd.read_csv("test_data.csv")

def clean_dataset(df):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip().str.lower()
    return df

train_data = clean_dataset(train_data)
test_data = clean_dataset(test_data)

X_train = train_data.drop('prognosis', axis=1)
y_train = train_data['prognosis']
X_test = test_data.drop('prognosis', axis=1)
y_test = test_data['prognosis']

# Feature selection
selector = SelectKBest(score_func=mutual_info_classif, k=120)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_features = X_train.columns[selector.get_support()].tolist()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Initialize and train multiple models
models = {
    'svm': SVC(kernel='rbf', probability=True, random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'decision_tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'naive_bayes': GaussianNB(),
    'knn': KNeighborsClassifier(n_neighbors=5)
}

# Train individual models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"{name} accuracy: {accuracy_score(y_test, y_pred)}")

# Create ensemble model
ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting='soft'
)
ensemble.fit(X_train_scaled, y_train)
ensemble_accuracy = accuracy_score(y_test, ensemble.predict(X_test_scaled))
print(f"Ensemble accuracy: {ensemble_accuracy}")

# Specialist mapping
disease_to_specialist = {
    "Fungal infection": "Dermatologist",
    "Allergy": "Allergist",
    "GERD": "Gastroenterologist",
    "Chronic cholestasis": "Gastroenterologist",
    "Drug Reaction": "Dermatologist",
    "Peptic ulcer disease": "Gastroenterologist",
    "AIDS": "Infectious Disease Specialist",
    "Diabetes": "Endocrinologist",
    "Gastroenteritis": "Gastroenterologist",
    "Bronchial Asthma": "Pulmonologist",
    "Hypertension": "Cardiologist",
    "Migraine": "Neurologist",
    "Cervical spondylosis": "Orthopedics",
    "Paralysis (brain hemorrhage)": "Neurologist",
    "Jaundice": "Gastroenterologist",
    "Malaria": "Infectious Disease Specialist",
    "Chicken pox": "Medicine",
    "Dengue": "Infectious Disease Specialist",
    "Typhoid": "Infectious Disease Specialist",
    "Hepatitis A": "Hepatologist",
    "Hepatitis B": "Hepatologist",
    "Hepatitis C": "Hepatologist",
    "Hepatitis D": "Hepatologist",
    "Hepatitis E": "Hepatologist",
    "Alcoholic hepatitis": "Gastroenterologist",
    "Tuberculosis": "Pulmonologist",
    "Common Cold": "Medicine",
    "Pneumonia": "Pulmonologist",
    "Dimorphic hemorrhoids(piles)": "Proctologist",
    "Heart attack": "Cardiologist",
    "Varicose veins": "General_Surgeon",
    "Hypothyroidism": "Endocrinologist",
    "Hyperthyroidism": "Endocrinologist",
    "Hypoglycemia": "Endocrinologist",
    "Osteoarthritis": "Orthopedics",
    "Arthritis": "Orthopedics",
    "(vertigo) Paroxysmal Positional Vertigo": "Neurologist",
    "Acne": "Dermatologist",
    "Urinary tract infection": "Urologist",
    "Psoriasis": "Dermatologist",
    "Impetigo": "Dermatologist"
}

allowed_tables = ["Cardiologist", "Dermatologist", "General_surgeon", "Endocrinologist",
                  "Gastroenterologist", "Medicine", "Haematology", "Neurologist", "Orthopedics", "Urologist"]

@app.route('/')
def home():
    return "Flask Backend is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_symptoms = data.get("symptoms", [])

        input_vector = {symptom: 0 for symptom in selected_features}
        for symptom in input_symptoms:
            if symptom in input_vector:
                input_vector[symptom] = 1

        input_df = pd.DataFrame([input_vector])
        input_scaled = scaler.transform(input_df)

        predicted_disease = ensemble.predict(input_scaled)[0]
        probabilities = ensemble.predict_proba(input_scaled)[0]
        
        specialist = disease_to_specialist.get(predicted_disease, "General Practitioner")

        return jsonify({
            "predicted_disease": predicted_disease,
            "recommended_specialist": specialist,
            "confidence": float(np.max(probabilities))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route("/api/signup", methods=["POST"])
# def signup():
#     data = request.get_json()
#     name = data["name"]
#     email = data["email"]
#     password = data["password"]

#     try:
#         conn = postgresql_pool.getconn()
#         cur = conn.cursor()
        
#         cur.execute("SELECT * FROM users WHERE email = %s", (email,))
#         if cur.fetchone():
#             return jsonify({"error": "Email already in use"}), 400

#         hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
#         cur.execute("INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s) RETURNING id, name, email",
#                     (name, email, hashed.decode()))
#         user = cur.fetchone()
#         conn.commit()
        
#         return jsonify({
#             "message": "User registered successfully", 
#             "user": {"id": user[0], "name": user[1], "email": user[2]}
#         })
#     except Exception as e:
#         if conn:
#             conn.rollback()
#         return jsonify({"error": str(e)}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             postgresql_pool.putconn(conn)

# @app.route("/api/login", methods=["POST"])
# def login():
#     data = request.get_json()
#     email = data["email"]
#     password = data["password"]

#     try:
#         conn = postgresql_pool.getconn()
#         cur = conn.cursor()
        
#         cur.execute("SELECT * FROM users WHERE email = %s", (email,))
#         user = cur.fetchone()

#         if not user or not bcrypt.checkpw(password.encode('utf-8'), user[3].encode()):
#             return jsonify({"error": "Invalid email or password"}), 401

#         return jsonify({
#             "message": "Login successful", 
#             "user": {"id": user[0], "name": user[1], "email": user[2]}
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#     finally:
#         if cur:
#             cur.close()
#         if conn:
#             postgresql_pool.putconn(conn)
from flask import jsonify, request
import bcrypt
import re
from datetime import datetime

def validate_email(email):
    """Basic email validation"""
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json()
    
    # Input validation
    if not all(key in data for key in ['name', 'email', 'password']):
        return jsonify({"error": "Missing name, email or password"}), 400
    
    name = data["name"].strip()
    email = data["email"].strip().lower()
    password = data["password"]
    
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    
    if not validate_email(email):
        return jsonify({"error": "Invalid email format"}), 400

    conn = None
    cur = None
    try:
        conn = postgresql_pool.getconn()
        cur = conn.cursor()
        
        # Check if email exists
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            return jsonify({"error": "Email already in use"}), 409
        
        # Store plain text password (INSECURE)
        cur.execute(
            """INSERT INTO users (name, email, password_hash, created_at) 
            VALUES (%s, %s, %s, %s) 
            RETURNING id, name, email""",
            (name, email, password, datetime.utcnow())  # Storing plain password
        )
        
        user = cur.fetchone()
        conn.commit()
        
        return jsonify({
            "status": "success",
            "user": {
                "id": user[0],
                "name": user[1],
                "email": user[2]
            }
        }), 201

    except Exception as e:
        if conn:
            conn.rollback()
        app.logger.error(f"Signup error: {str(e)}")
        return jsonify({"error": "Registration failed"}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            postgresql_pool.putconn(conn)

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    
    # Validation
    if not all(key in data for key in ['email', 'password']):
        return jsonify({"error": "Email and password required"}), 400
    
    email = data["email"].strip().lower()
    password = data["password"].strip()  # Added strip()
    
    conn = None
    cur = None
    try:
        conn = postgresql_pool.getconn()
        cur = conn.cursor()
        
        # Verify column name matches your database
        cur.execute("SELECT id, name, email, password_hash FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        
        if not user:
            return jsonify({"error": "Invalid credentials"}), 401
        
        # Direct comparison (for plain text passwords)
        if password != user[3]:
            return jsonify({"error": "Invalid credentials"}), 401
        
        # # Update last login
        # cur.execute("UPDATE users SET last_login = %s WHERE id = %s", 
        #            (datetime.utcnow(), user[0]))
        # conn.commit()
        
        return jsonify({
            "status": "success",
            "user": {
                "id": user[0],
                "name": user[1],
                "email": user[2]
            }
        })

    except Exception as e:
        app.logger.error(f"Login error: {str(e)}")
        return jsonify({"error": "Login failed"}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            postgresql_pool.putconn(conn)
@app.route("/api/opd/<opd_name>", methods=["GET"])
def get_opd(opd_name):
    if opd_name not in allowed_tables:
        return jsonify({"error": "Invalid OPD name"}), 400

    try:
        conn = postgresql_pool.getconn()
        cur = conn.cursor()

        cur.execute(f"SELECT * FROM {opd_name}")
        doctors = cur.fetchall()
        doctor_columns = [desc[0] for desc in cur.description]

        cur.execute("SELECT opd_number, address FROM opd_location WHERE opd_name = %s", (opd_name,))
        location = cur.fetchone()
        opd_number = location[0] if location else "N/A"
        address = location[1] if location else "N/A"

        doctor_list = [dict(zip(doctor_columns, row)) for row in doctors]

        for doc in doctor_list:
            doc["opd_number"] = opd_number
            doc["address"] = address

        return jsonify(doctor_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            postgresql_pool.putconn(conn)

@app.route("/api/store-prediction", methods=["POST"])
def store_prediction():
    data = request.get_json()
    required = ["user_id", "disease", "specialist"]
    if not all(k in data for k in required):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        conn = postgresql_pool.getconn()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO disease_prediction 
            (user_id, symptom1, symptom2, symptom3, symptom4, symptom5, disease, specialist, predicted_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            RETURNING *;
        """, (data["user_id"], data.get("symptom1"), data.get("symptom2"), data.get("symptom3"),
              data.get("symptom4"), data.get("symptom5"), data["disease"], data["specialist"]))

        conn.commit()
        return jsonify({"message": "Prediction stored"})
    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            postgresql_pool.putconn(conn)

@app.route("/api/user-history/<int:user_id>", methods=["GET"])
def user_history(user_id):
    try:
        conn = postgresql_pool.getconn()
        cur = conn.cursor()

        cur.execute("""
            SELECT symptom1, symptom2, symptom3, symptom4, symptom5, disease, specialist, predicted_at
            FROM disease_prediction
            WHERE user_id = %s
            ORDER BY predicted_at DESC;
        """, (user_id,))
        rows = cur.fetchall()
        if not rows:
            return jsonify({"message": "No history found for this user."}), 404

        return jsonify([
            {
                "symptom1": row[0], "symptom2": row[1], "symptom3": row[2],
                "symptom4": row[3], "symptom5": row[4],
                "disease": row[5], "specialist": row[6], "predicted_at": row[7].isoformat()
            } for row in rows
        ])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            postgresql_pool.putconn(conn)

@app.route("/api/all-history", methods=["GET"])
def all_history():
    try:
        conn = postgresql_pool.getconn()
        cur = conn.cursor()

        cur.execute("""
            SELECT 
                dp.id, u.id, u.name, u.email,
                dp.symptom1, dp.symptom2, dp.symptom3, dp.symptom4, dp.symptom5,
                dp.disease, dp.specialist, dp.predicted_at
            FROM disease_prediction dp
            JOIN users u ON dp.user_id = u.id
            ORDER BY dp.predicted_at DESC;
        """)
        rows = cur.fetchall()
        if not rows:
            return jsonify({"message": "No history records found."}), 404

        return jsonify([
            {
                "prediction_id": r[0], "user_id": r[1], "user_name": r[2], "user_email": r[3],
                "symptom1": r[4], "symptom2": r[5], "symptom3": r[6],
                "symptom4": r[7], "symptom5": r[8],
                "disease": r[9], "specialist": r[10], "predicted_at": r[11].isoformat()
            } for r in rows
        ])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            postgresql_pool.putconn(conn)



@app.route('/api/check-db-connection')
def check_db_connection():
    try:
        conn = postgresql_pool.getconn()
        cur = conn.cursor()
        
        # Test query
        cur.execute("SELECT 1")
        result = cur.fetchone()
        
        return jsonify({
            "status": "success",
            "message": "Database connection established",
            "test_result": result[0]
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Database connection failed",
            "error": str(e)
        }), 500
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): postgresql_pool.putconn(conn)


@app.route('/api/temp', methods=['GET'])
def get_temp_data():
    try:
        conn = postgresql_pool.getconn()
        cur = conn.cursor()
        
        # Fetch all records from temp table
        cur.execute("SELECT id, name FROM temp ORDER BY id")
        temp_data = cur.fetchall()
        
        # Convert to list of dictionaries
        result = [{"id": row[0], "name": row[1]} for row in temp_data]
        
        return jsonify({
            "status": "success",
            "data": result,
            "count": len(result)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to fetch temp data",
            "error": str(e)
        }), 500
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): postgresql_pool.putconn(conn)
if __name__ == "__main__":
    app.run(debug=True, port=5000)