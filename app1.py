import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from poliastro.bodies import Sun, Earth
from poliastro.twobody import Orbit
from astropy import units as u
from astropy.time import Time
import tempfile
import os
import joblib
from collections import Counter
from pathlib import Path
import base64

# =====================================================
# Background Image Functions & Setup
# =====================================================
@st.cache_resource
def get_image_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Set base and image directories
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "data"

# Load images (update filenames as needed)
img_main = get_image_as_base64(IMAGE_DIR / "2773642.jpg")
img_sidebar = get_image_as_base64(IMAGE_DIR / "for1.jpg")

# Apply CSS for background images
st.markdown(f"""
    <style>
    .main {{
        background-image: url("data:image/png;base64,{img_main}");
        background-size: cover;
    }}
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{img_main}");
        background-size: cover;
        background-position: center;
    }}
    [data-testid="stSidebarContent"] {{
        background-image: url("data:image/png;base64,{img_sidebar}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# Sidebar Navigation
# =====================================================
st.title("Asteroid Analysis Tool")
page = st.sidebar.radio("Select Page", ["Home", "Classification", "Prediction"])

# =====================================================
# Home Page
# =====================================================
if page == "Home":
    st.header("Welcome to the Asteroid Analysis Tool")
    st.markdown("""
    This application provides two main functionalities:
    
    **1. Classification:**  
    Classify an asteroid as hazardous or non-hazardous based on manually entered orbital parameters.
      
    **2. Prediction:**  
    Predict the future path of an asteroid using uploaded observational data and a hybrid (physics + ML) approach.
    
    Use the sidebar to select the desired function.
    """)

# =====================================================
# Classification Page
# =====================================================
elif page == "Classification":
    st.header("Asteroid Hazard Classification")
    st.markdown("Enter asteroid parameters for classification:")

    # Input fields for classification (manually entered values)
    neo = st.selectbox("Near Earth Object (neo)", [1, 0])
    H = st.number_input("Absolute Magnitude (H)", value=17.5)
    e = st.number_input("Eccentricity (e)", value=0.62)
    a = st.number_input("Semi-major Axis (a)", value=1.95)
    q = st.number_input("Perihelion Distance (q)", value=0.85)
    i = st.number_input("Inclination (i)", value=15.4)
    om = st.number_input("Longitude of Ascending Node (om)", value=50.2)
    w = st.number_input("Argument of Periapsis (w)", value=210.3)
    n = st.number_input("Mean Motion (n)", value=0.45)
    per = st.number_input("Orbital Period (per)", value=900)
    moid = st.number_input("Earth MOID (moid)", value=0.025)
    moid_ld = st.number_input("Earth MOID in Lunar Distance (moid_ld)", value=9.7)
    moid_jup = st.number_input("Jupiter MOID (moid_jup)", value=1.5)
    condition_code = st.number_input("Condition Code", value=1)

    # Prepare the input DataFrame for classification
    df_hazardous = pd.DataFrame([{
        'neo': neo,
        'H': H,
        'e': e,
        'a': a,
        'q': q,
        'i': i,
        'om': om,
        'w': w,
        'n': n,
        'per': per,
        'moid': moid,
        'moid_ld': moid_ld,
        'moid_jup': moid_jup,
        'condition_code': condition_code
    }])
    st.write("### Input Data:")
    st.write(df_hazardous)

    if st.button("Classify Asteroid"):
        try:
            # Set model paths (update these if needed)
            rf_model_path = os.path.join("models", "RF_model (1).pkl")
            xgb_model_path = os.path.join("models", "xgboost_asteroid_model.pkl")

            # Load the two classification models using joblib
            RF_model = joblib.load(rf_model_path)
            xgb_model = joblib.load(xgb_model_path)

            # Get predictions from both models
            rf_prediction = RF_model.predict(df_hazardous)[0]
            xgb_prediction = xgb_model.predict(df_hazardous)[0]

            # Ensemble approach: average the predictions and round to get final label
            ensemble_prediction = round((rf_prediction + xgb_prediction) / 2)

            if ensemble_prediction == 1:
                st.success("The asteroid is classified as **Hazardous**.")
            else:
                st.info("The asteroid is classified as **Non-Hazardous**.")
        except FileNotFoundError:
            st.error("Model files not found. Please ensure 'RF_model (1).pkl' and 'xgboost_asteroid_model.pkl' are in the 'models' subdirectory.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# =====================================================
# Prediction Page
# =====================================================
elif page == "Prediction":
    st.header("Asteroid Path Prediction and Simulation")
    st.markdown("""
    This section performs trajectory prediction using a physics-based orbit propagation model corrected by an ML model.
    
    **Visualizations:**
    - **Animated Simulation:** Displays the asteroid’s trajectory over time (physics-based prediction).
    - **Static 3D Comparison:** Shows a comparison between the physics prediction (cyan), the hybrid prediction (red), and Earth’s orbit (blue).
    
    Additional graphs show the asteroid's distance from the Sun and the magnitude of the ML correction over time.
    
    Press **Run Prediction** to start.
    """)

    uploaded_csv = st.file_uploader("Upload Asteroid Observation CSV", type="csv")
    num_future = st.number_input("Number of Future Steps", min_value=1, value=500)

    # Path for the hybrid correction model (hardcoded)
    model_path = "models/hybrid_correction_model.h5"

    if st.button("Run Prediction"):
        if uploaded_csv is not None:
            # ===============================================
            # 1. Load Asteroid Observation Data from CSV
            # ===============================================
            col_names = [
                "dRA*cosD", "d(DEC)/dt", "hEcl-Lon", "hEcl-Lat", "r", "rdot", "delta",
                "deldot", "VmagSn", "VmagOb", "S-O-T", "PlAng", "aid", "RA_Hours",
                "RA_Minutes", "RA_Seconds", "DEC_Degrees", "DEC_Minutes", "DEC_Seconds", "Date"
            ]
            df = pd.read_csv(uploaded_csv, header=None, names=col_names, skiprows=1, delimiter=',')
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
            df.sort_values('Date', inplace=True)

            # ===============================================
            # 2. Convert Spherical to Cartesian Coordinates
            # ===============================================
            def sph_to_cart(r, lon, lat):
                lon_rad = np.deg2rad(lon)
                lat_rad = np.deg2rad(lat)
                x = r * np.cos(lat_rad) * np.cos(lon_rad)
                y = r * np.cos(lat_rad) * np.sin(lon_rad)
                z = r * np.sin(lat_rad)
                return np.array([x, y, z])

            cart_coords = df.apply(lambda row: sph_to_cart(row['r'], row['hEcl-Lon'], row['hEcl-Lat']), axis=1)
            df[['x', 'y', 'z']] = pd.DataFrame(cart_coords.tolist(), index=df.index)

            # ===============================================
            # 3. Estimate Initial Velocity & Create Physics-Based Orbit
            # ===============================================
            dt = (df['Date'].iloc[1] - df['Date'].iloc[0]).total_seconds()
            initial_pos = np.array([df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0]])
            second_pos = np.array([df['x'].iloc[1], df['y'].iloc[1], df['z'].iloc[1]])
            initial_vel = (second_pos - initial_pos) / dt  # in AU per second
            initial_time = Time(df['Date'].iloc[0])
            orbit = Orbit.from_vectors(Sun, initial_pos * u.AU, initial_vel * u.AU / u.s, epoch=initial_time)

            # ===============================================
            # 4. Compute Physics-Based Predictions on Training Data
            # ===============================================
            physics_predictions = []
            time_deltas = []
            for obs_time in df['Date']:
                delta_sec = (obs_time - df['Date'].iloc[0]).total_seconds()
                time_deltas.append(delta_sec)
                propagated_orbit = orbit.propagate(delta_sec * u.s)
                r_vec = propagated_orbit.r.to(u.AU).value
                physics_predictions.append(r_vec)
            physics_predictions = np.array(physics_predictions)
            observed_positions = df[['x', 'y', 'z']].values
            residuals = observed_positions - physics_predictions

            # ===============================================
            # 5. Prepare Data for ML Correction
            # ===============================================
            X_ml = np.hstack([physics_predictions, np.array(time_deltas).reshape(-1, 1)])
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X_ml_scaled = scaler_X.fit_transform(X_ml)
            scaler_y.fit(residuals)

            # ===============================================
            # 6. Load Hybrid Correction Model (Physics + ML)
            # ===============================================
            def mse(y_true, y_pred):
                return tf.reduce_mean(tf.square(y_true - y_pred))
            model_ml = load_model(model_path, custom_objects={'mse': mse})

            # ===============================================
            # 7. Compute Future Predictions
            # ===============================================
            physics_future = []      # Physics-based future positions
            ml_predictions = []      # ML predicted corrections (residuals)
            hybrid_predictions = []  # Combined prediction: physics + ML correction

            avg_interval = np.mean(np.diff(df['Date']).astype('timedelta64[s]').astype(np.float64))
            last_time = df['Date'].iloc[-1]
            future_times = []
            for i in range(1, num_future + 1):
                future_time = last_time + pd.to_timedelta(i * avg_interval, unit='s')
                future_times.append(future_time)
                delta_sec = (future_time - df['Date'].iloc[0]).total_seconds()
                propagated_orbit = orbit.propagate(delta_sec * u.s)
                phys_pos = propagated_orbit.r.to(u.AU).value
                physics_future.append(phys_pos)

                ml_input = np.hstack([phys_pos, delta_sec]).reshape(1, -1)
                ml_input_scaled = scaler_X.transform(ml_input)
                correction_scaled = model_ml.predict(ml_input_scaled)
                correction = scaler_y.inverse_transform(correction_scaled).flatten()
                ml_predictions.append(correction)

                hybrid_pos = phys_pos + correction
                hybrid_predictions.append(hybrid_pos)
            physics_future = np.array(physics_future)
            ml_predictions = np.array(ml_predictions)
            hybrid_predictions = np.array(hybrid_predictions)

            # ===============================================
            # 8. Propagate Earth Orbit for the Future Span
            # ===============================================
            earth_orbit = Orbit.from_classical(
                Sun,
                1 * u.AU,
                0.0167 * u.one,
                0 * u.deg,
                0 * u.deg,
                102.9372 * u.deg,
                0 * u.deg,
                epoch=Time(last_time)
            )
            future_days = np.linspace(0, (num_future * avg_interval) / (24*3600), num_future)
            last_time_astropy = Time(last_time)
            times_earth = Time(last_time_astropy.jd + future_days, format='jd')
            rr_earth = np.array([
                earth_orbit.propagate((t - Time(last_time)).to(u.day).value * u.day).r.to(u.AU).value
                for t in times_earth
            ])

            # ===============================================
            # 9a. Animated Simulation (Physics-Based Only)
            # ===============================================
            n_points = num_future
            init_ast_pos = physics_future[0]
            init_earth_pos = rr_earth[0]
            anim_data = [
                go.Scatter3d(
                    x=[init_ast_pos[0]], y=[init_ast_pos[1]], z=[init_ast_pos[2]],
                    mode='markers',
                    marker=dict(size=5, color='cyan'),
                    name='Asteroid (Physics)'
                ),
                go.Scatter3d(
                    x=[init_earth_pos[0]], y=[init_earth_pos[1]], z=[init_earth_pos[2]],
                    mode='markers',
                    marker=dict(size=5, color='blue'),
                    name='Earth'
                ),
                go.Scatter3d(
                    x=physics_future[:1, 0], y=physics_future[:1, 1], z=physics_future[:1, 2],
                    mode='lines',
                    line=dict(color='cyan', width=2),
                    name='Asteroid Path (Physics)'
                ),
                go.Scatter3d(
                    x=rr_earth[:1, 0], y=rr_earth[:1, 1], z=rr_earth[:1, 2],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Earth Path'
                ),
                go.Scatter3d(
                    x=[0], y=[0], z=[0],
                    mode='markers',
                    marker=dict(size=8, color='yellow'),
                    name='Sun'
                )
            ]
            anim_frames = []
            for i in range(n_points):
                frame = go.Frame(
                    data=[
                        go.Scatter3d(
                            x=[physics_future[i, 0]], y=[physics_future[i, 1]], z=[physics_future[i, 2]],
                            mode='markers',
                            marker=dict(size=5, color='cyan'),
                            name='Asteroid (Physics)'
                        ),
                        go.Scatter3d(
                            x=[rr_earth[i, 0]], y=[rr_earth[i, 1]], z=[rr_earth[i, 2]],
                            mode='markers',
                            marker=dict(size=5, color='blue'),
                            name='Earth'
                        ),
                        go.Scatter3d(
                            x=physics_future[:i+1, 0], y=physics_future[:i+1, 1], z=physics_future[:i+1, 2],
                            mode='lines',
                            line=dict(color='cyan', width=2),
                            name='Asteroid Path (Physics)'
                        ),
                        go.Scatter3d(
                            x=rr_earth[:i+1, 0], y=rr_earth[:i+1, 1], z=rr_earth[:i+1, 2],
                            mode='lines',
                            line=dict(color='blue', width=2),
                            name='Earth Path'
                        ),
                        go.Scatter3d(
                            x=[0], y=[0], z=[0],
                            mode='markers',
                            marker=dict(size=8, color='yellow'),
                            name='Sun'
                        )
                    ],
                    name=f'frame{i}'
                )
                anim_frames.append(frame)
            anim_fig = go.Figure(
                data=anim_data,
                layout=go.Layout(
                    title='Animated Asteroid Simulation',
                    paper_bgcolor="black",
                    plot_bgcolor="black",
                    font=dict(color="white"),
                    scene=dict(
                        xaxis_title='X (AU)',
                        yaxis_title='Y (AU)',
                        zaxis_title='Z (AU)',
                        bgcolor="black",
                        xaxis=dict(backgroundcolor="black", gridcolor="grey", zerolinecolor="grey", color="white"),
                        yaxis=dict(backgroundcolor="black", gridcolor="grey", zerolinecolor="grey", color="white"),
                        zaxis=dict(backgroundcolor="black", gridcolor="grey", zerolinecolor="grey", color="white")
                    ),
                    updatemenus=[{
                        "buttons": [
                            {
                                "args": [None, {"frame": {"duration": 100, "redraw": True},
                                                "fromcurrent": True, "transition": {"duration": 0}}],
                                "label": "Play",
                                "method": "animate"
                            },
                            {
                                "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                                  "mode": "immediate",
                                                  "transition": {"duration": 0}}],
                                "label": "Pause",
                                "method": "animate"
                            }
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 87},
                        "showactive": False,
                        "type": "buttons",
                        "x": 0.1,
                        "xanchor": "right",
                        "y": 0,
                        "yanchor": "top"
                    }],
                    sliders=[{
                        "steps": [
                            {
                                "args": [[f'frame{k}'], {"frame": {"duration": 0, "redraw": True},
                                                          "mode": "immediate",
                                                          "transition": {"duration": 0}}],
                                "label": f"{k}",
                                "method": "animate"
                            } for k in range(n_points)
                        ],
                        "transition": {"duration": 0},
                        "x": 0.1,
                        "y": 0,
                        "currentvalue": {"font": {"size": 12}, "prefix": "Frame: ", "visible": True, "xanchor": "right"},
                        "len": 0.9
                    }]
                ),
                frames=anim_frames
            )
            st.plotly_chart(anim_fig, use_container_width=True)

            # ===============================================
            # 9b. Static 3D Plot: Comparison of Predictions
            # ===============================================
            trace_physics = go.Scatter3d(
                x=physics_future[:, 0],
                y=physics_future[:, 1],
                z=physics_future[:, 2],
                mode='lines+markers',
                line=dict(color='cyan', width=3),
                marker=dict(size=3, color='cyan'),
                name='Physics Prediction'
            )
            trace_hybrid = go.Scatter3d(
                x=hybrid_predictions[:, 0],
                y=hybrid_predictions[:, 1],
                z=hybrid_predictions[:, 2],
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=3, color='red'),
                name='Hybrid Prediction'
            )
            trace_earth = go.Scatter3d(
                x=rr_earth[:, 0],
                y=rr_earth[:, 1],
                z=rr_earth[:, 2],
                mode='lines+markers',
                line=dict(color='blue', width=3),
                marker=dict(size=3, color='blue'),
                name='Earth Orbit'
            )
            correction_lines = []
            for i in range(num_future):
                correction_lines.append(
                    go.Scatter3d(
                        x=[physics_future[i, 0], hybrid_predictions[i, 0]],
                        y=[physics_future[i, 1], hybrid_predictions[i, 1]],
                        z=[physics_future[i, 2], hybrid_predictions[i, 2]],
                        mode='lines',
                        line=dict(color='green', width=2),
                        showlegend=False
                    )
                )
            static_data = [trace_physics, trace_hybrid, trace_earth] + correction_lines
            static_layout = go.Layout(
                title='Static 3D Prediction Comparison',
                paper_bgcolor="black",
                plot_bgcolor="black",
                font=dict(color="white"),
                scene=dict(
                    xaxis_title='X (AU)',
                    yaxis_title='Y (AU)',
                    zaxis_title='Z (AU)',
                    bgcolor="black",
                    xaxis=dict(backgroundcolor="black", gridcolor="grey", zerolinecolor="grey", color="white"),
                    yaxis=dict(backgroundcolor="black", gridcolor="grey", zerolinecolor="grey", color="white"),
                    zaxis=dict(backgroundcolor="black", gridcolor="grey", zerolinecolor="grey", color="white")
                )
            )
            static_fig = go.Figure(data=static_data, layout=static_layout)
            st.plotly_chart(static_fig, use_container_width=True)

            # ===============================================
            # 10. Additional Graphs & Explanations
            # ===============================================
            st.markdown("### Additional Analysis")
            st.markdown("""
            **Distance Over Time:**  
            This graph shows the distance (from the Sun, assumed at [0,0,0]) of the asteroid over time for the physics-based and hybrid predictions.  
            **ML Correction Magnitude:**  
            This graph displays the magnitude of the ML correction (i.e. the difference between the hybrid and physics-based predictions) over time.
            """)
            future_days = np.array([(ft - df['Date'].iloc[0]).total_seconds() / 86400.0 for ft in future_times])
            physics_dist = np.linalg.norm(physics_future, axis=1)
            hybrid_dist = np.linalg.norm(hybrid_predictions, axis=1)
            correction_mag = np.linalg.norm(ml_predictions, axis=1)
            dist_fig = go.Figure()
            dist_fig.add_trace(go.Scatter(x=future_days, y=physics_dist, mode='lines+markers',
                                          line=dict(color='cyan', width=3), name='Physics Distance'))
            dist_fig.add_trace(go.Scatter(x=future_days, y=hybrid_dist, mode='lines+markers',
                                          line=dict(color='red', width=3), name='Hybrid Distance'))
            dist_fig.update_layout(title="Asteroid Distance from Sun Over Time",
                                    xaxis_title="Time (days)",
                                    yaxis_title="Distance (AU)",
                                    paper_bgcolor="black",
                                    plot_bgcolor="black",
                                    font=dict(color="white"))
            st.plotly_chart(dist_fig, use_container_width=True)
            corr_fig = go.Figure()
            corr_fig.add_trace(go.Scatter(x=future_days, y=correction_mag, mode='lines+markers',
                                          line=dict(color='green', width=3), name='ML Correction Magnitude'))
            corr_fig.update_layout(title="ML Correction Magnitude Over Time",
                                    xaxis_title="Time (days)",
                                    yaxis_title="Correction Magnitude (AU)",
                                    paper_bgcolor="black",
                                    plot_bgcolor="black",
                                    font=dict(color="white"))
            st.plotly_chart(corr_fig, use_container_width=True)
        else:
            st.info("Please upload a CSV file to see the prediction and simulation.")
