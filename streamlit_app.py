import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

st.set_page_config(page_title="RentPredict", page_icon="🏠", layout="wide")

CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "default_data.csv")

def load_data():
    if "data" not in st.session_state:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE, encoding="utf-8")
        else:
            df = pd.DataFrame(columns=["name", "distance", "room_size", "convenience", "fitness", "room_condition", "price"])
        st.session_state.data = df
    return st.session_state.data

def save_data(df):
    st.session_state.data = df

st.markdown("""
<style>
    .main-title { text-align: center; font-size: 2.5rem; font-weight: 800; margin-bottom: 0; }
    .sub-title { text-align: center; color: #94a3b8; margin-bottom: 2rem; }
    .metric-card { background: #1e293b; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #334155; }
    .metric-value { font-size: 2rem; font-weight: 800; }
    .metric-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; }
    .predict-box { background: linear-gradient(135deg, #1e1b4b, #312e81); border-radius: 16px; padding: 40px; text-align: center; border: 1px solid #4338ca; }
    .predict-price { font-size: 3rem; font-weight: 900; color: #f59e0b; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🏠 RentPredict</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">ทำนายราคาค่าเช่าหอพักข้างมหาวิทยาลัย — Multiple Regression Model</p>', unsafe_allow_html=True)

tab_home, tab_data, tab_train, tab_predict = st.tabs(["🏠 Home", "📝 กรอกข้อมูล", "🧠 Train Model", "🔮 ทำนาย"])

with tab_home:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📝 กรอกข้อมูล")
        st.write("เพิ่มข้อมูลหอพักที่สำรวจมา เช่น ระยะทาง ขนาดห้อง สภาพห้อง และราคาค่าเช่า")
    with col2:
        st.markdown("### 🧠 Train Model")
        st.write("สร้างโมเดล Multiple Regression แบ่ง Train/Test 80:20 พร้อมประเมินผล R², MAE, MSE, RMSE")
    with col3:
        st.markdown("### 🔮 ทำนายราคา")
        st.write("กรอกข้อมูลหอพักที่สนใจ แล้วให้โมเดลทำนายราคาค่าเช่าที่เหมาะสม")

    st.markdown("---")
    st.markdown("### 📊 Features ที่ใช้ในโมเดล")
    features_df = pd.DataFrame({
        "Feature": ["Distance", "Room Size", "Convenience", "Fitness", "Room Condition"],
        "คำอธิบาย": ["ระยะทางจากมหาวิทยาลัย", "ขนาดห้อง", "การเดินทางสะดวก", "มีฟิตเนสหรือไม่", "สภาพห้อง"],
        "ประเภท": ["กม.", "ตร.ม.", "0 = ยาก, 1 = ง่าย/มีวิน", "0 = ไม่มี, 1 = มี", "คะแนน 1-5"]
    })
    st.dataframe(features_df, use_container_width=True, hide_index=True)

with tab_data:
    st.markdown("### ➕ เพิ่มข้อมูลหอพัก")
    with st.form("add_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("ชื่อหอพัก", placeholder="เช่น หอพักสุขใจ")
            distance = st.number_input("ระยะทาง (กม.)", min_value=0.0, step=0.1, format="%.1f")
            room_size = st.number_input("ขนาดห้อง (ตร.ม.)", min_value=1.0, step=0.5, format="%.1f")
            price = st.number_input("ราคาเช่า/เดือน (บาท)", min_value=0, step=100)
        with col2:
            convenience = st.selectbox("การเดินทาง", options=[1, 0], format_func=lambda x: "🟢 ง่าย / มีวิน" if x == 1 else "🔴 ยาก")
            fitness = st.selectbox("ฟิตเนส", options=[0, 1], format_func=lambda x: "✅ มี" if x == 1 else "❌ ไม่มี")
            room_condition = st.selectbox("สภาพห้อง (1-5)", options=[5, 4, 3, 2, 1], format_func=lambda x: f"⭐ {x}")

        submitted = st.form_submit_button("➕ เพิ่มข้อมูล", use_container_width=True)
        if submitted and name:
            df = load_data()
            new_row = pd.DataFrame([{
                "name": name, "distance": distance, "room_size": room_size,
                "convenience": convenience, "fitness": fitness,
                "room_condition": room_condition, "price": price
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            save_data(df)
            st.success(f'เพิ่ม "{name}" สำเร็จ!')
            st.rerun()

    st.markdown("---")
    st.markdown("### 📋 รายการข้อมูลหอพัก")
    df = load_data()
    st.info(f"ข้อมูลทั้งหมด **{len(df)}** รายการ")

    if len(df) > 0:
        display_df = df.copy()
        display_df.index = range(1, len(display_df) + 1)
        display_df.columns = ["ชื่อหอพัก", "ระยะทาง (กม.)", "ขนาดห้อง (ตร.ม.)", "การเดินทาง", "ฟิตเนส", "สภาพห้อง", "ราคา/เดือน"]
        st.dataframe(display_df, use_container_width=True)

        st.markdown("#### 🗑️ ลบข้อมูล")
        del_idx = st.number_input("ลำดับที่ต้องการลบ", min_value=1, max_value=len(df), step=1)
        if st.button("🗑️ ลบรายการนี้"):
            df = df.drop(df.index[del_idx - 1]).reset_index(drop=True)
            save_data(df)
            st.success("ลบข้อมูลสำเร็จ")
            st.rerun()

with tab_train:
    st.markdown("### 🧠 Train Multiple Regression Model")
    st.write("กดปุ่มด้านล่างเพื่อ Train model ด้วย Linear Regression (Multiple Regression)")
    st.write("ใช้ Train/Test Split 80:20 และประเมินผลด้วย R², MAE, MSE, RMSE")

    if st.button("🚀 Train Model", use_container_width=True):
        df = load_data()
        if len(df) < 5:
            st.error("ต้องมีข้อมูลอย่างน้อย 5 รายการ")
        else:
            feature_cols = ["distance", "room_size", "convenience", "fitness", "room_condition"]
            X = df[feature_cols].values.astype(float)
            y = df["price"].values.astype(float)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred) if len(y_test) > 1 else float("nan")
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)

            st.session_state.model = model
            st.session_state.model_info = {
                "train_r2": train_r2, "test_r2": test_r2,
                "test_mae": test_mae, "test_mse": test_mse, "test_rmse": test_rmse,
                "train_size": len(X_train), "test_size": len(X_test),
                "total": len(df), "feature_cols": feature_cols,
                "coefficients": dict(zip(feature_cols, model.coef_)),
                "intercept": model.intercept_,
                "test_actual": y_test.tolist(), "test_pred": y_test_pred.tolist()
            }
            st.success("Train Model สำเร็จ!")

    if "model_info" in st.session_state:
        info = st.session_state.model_info
        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ข้อมูลทั้งหมด", info["total"])
        c2.metric("Train Set (80%)", info["train_size"])
        c3.metric("Test Set (20%)", info["test_size"])
        c4.metric("Train R²", f"{info['train_r2']:.4f}")

        st.markdown("---")
        st.markdown("#### 📐 Regression Equation")
        eq_parts = [f"{v:+.4f} × {k}" for k, v in info["coefficients"].items()]
        st.code(f"Price = {info['intercept']:.4f} {' '.join(eq_parts)}", language=None)

        st.markdown("#### 📊 Coefficients")
        feature_labels = {
            "distance": "ระยะทาง (กม.)", "room_size": "ขนาดห้อง (ตร.ม.)",
            "convenience": "การเดินทาง", "fitness": "ฟิตเนส", "room_condition": "สภาพห้อง"
        }
        coeff_df = pd.DataFrame([
            {"Feature": feature_labels.get(k, k), "Coefficient": f"{v:+.4f}"}
            for k, v in info["coefficients"].items()
        ])
        st.dataframe(coeff_df, use_container_width=True, hide_index=True)

        st.markdown("#### 📈 Evaluation Metrics")
        m1, m2, m3, m4 = st.columns(4)
        test_r2_val = info["test_r2"]
        m1.metric("R² Score", f"{test_r2_val:.4f}" if not np.isnan(test_r2_val) else "N/A")
        m2.metric("MAE", f"฿{info['test_mae']:,.2f}")
        m3.metric("MSE", f"฿{info['test_mse']:,.2f}")
        m4.metric("RMSE", f"฿{info['test_rmse']:,.2f}")

        if info["test_actual"]:
            st.markdown("#### 📉 Actual vs Predicted (Test Set)")
            chart_df = pd.DataFrame({
                "Actual": info["test_actual"],
                "Predicted": [round(v, 2) for v in info["test_pred"]]
            })
            chart_df.index = [f"Test {i+1}" for i in range(len(chart_df))]
            st.bar_chart(chart_df)

with tab_predict:
    st.markdown("### 🔮 ทำนายราคาค่าเช่า")
    st.write("กรอกข้อมูลหอพักที่ต้องการทำนาย แล้วกดปุ่มทำนาย")

    col1, col2 = st.columns(2)
    with col1:
        p_distance = st.number_input("ระยะทาง (กม.)", min_value=0.0, step=0.1, format="%.1f", key="pred_dist")
        p_room_size = st.number_input("ขนาดห้อง (ตร.ม.)", min_value=1.0, step=0.5, format="%.1f", key="pred_room")
        p_convenience = st.selectbox("การเดินทาง", options=[1, 0], format_func=lambda x: "🟢 ง่าย / มีวิน" if x == 1 else "🔴 ยาก", key="pred_conv")
    with col2:
        p_fitness = st.selectbox("ฟิตเนส", options=[0, 1], format_func=lambda x: "✅ มี" if x == 1 else "❌ ไม่มี", key="pred_fit")
        p_room_condition = st.selectbox("สภาพห้อง (1-5)", options=[5, 4, 3, 2, 1], format_func=lambda x: f"⭐ {x}", key="pred_cond")

    if st.button("🔮 ทำนายราคา", use_container_width=True):
        if "model" not in st.session_state:
            st.error("ยังไม่ได้ train model กรุณา train ก่อน")
        else:
            features = [p_distance, p_room_size, p_convenience, p_fitness, p_room_condition]
            prediction = st.session_state.model.predict([features])[0]
            prediction = max(0, round(prediction, 2))

            st.markdown("---")
            st.markdown(f"""
            <div class="predict-box">
                <p style="color:#94a3b8; font-size:1rem;">ราคาค่าเช่าที่ทำนาย</p>
                <p class="predict-price">฿{prediction:,.2f}</p>
                <p style="color:#94a3b8;">บาท / เดือน</p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
