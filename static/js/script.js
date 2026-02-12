const API_BASE = "/api";

function switchTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(tc => tc.classList.remove("active"));
  document.querySelector(`[data-tab="${tabName}"]`).classList.add("active");
  document.getElementById(`tab-${tabName}`).classList.add("active");
  if (tabName === "data") loadData();
  if (tabName === "train") loadModelInfo();
}

function showToast(message, type = "success") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  const icons = { success: "✅", error: "❌", info: "ℹ️" };
  toast.innerHTML = `<span>${icons[type] || ""}</span><span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(40px)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

async function loadData() {
  try {
    const res = await fetch(`${API_BASE}/data`);
    const data = await res.json();
    renderDataTable(data);
  } catch (e) {
    showToast("ไม่สามารถเชื่อมต่อ Server ได้", "error");
  }
}

function renderDataTable(data) {
  const tbody = document.getElementById("data-tbody");
  const countEl = document.getElementById("data-count");
  countEl.innerHTML = `ข้อมูลทั้งหมด <span class="count-num">${data.length}</span> รายการ`;

  if (data.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;color:var(--text-muted);padding:30px;">ยังไม่มีข้อมูล</td></tr>`;
    return;
  }

  tbody.innerHTML = data.map((d, i) => `
    <tr>
      <td>${i + 1}</td>
      <td class="name-cell">${d.name}</td>
      <td>${d.distance} กม.</td>
      <td>${d.room_size} ตร.ม.</td>
      <td>${d.convenience == 1 ? "🟢 ง่าย/มีวิน" : "🔴 ยาก"}</td>
      <td>${d.fitness == 1 ? "✅ มี" : "❌ ไม่มี"}</td>
      <td>⭐ ${d.room_condition}/5</td>
      <td class="price-cell">฿${Number(d.price).toLocaleString()}</td>
      <td><button class="btn btn-danger" onclick="deleteData('${d.id}')">🗑️ ลบ</button></td>
    </tr>
  `).join("");
}

async function addData(e) {
  e.preventDefault();
  const form = e.target;
  const entry = {
    name: form.name.value.trim(),
    distance: parseFloat(form.distance.value),
    room_size: parseFloat(form.room_size.value),
    convenience: parseInt(form.convenience.value),
    fitness: parseInt(form.fitness.value),
    room_condition: parseInt(form.room_condition.value),
    price: parseFloat(form.price.value),
  };

  if (!entry.name) { showToast("กรุณากรอกชื่อหอพัก", "error"); return; }

  try {
    const res = await fetch(`${API_BASE}/data`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(entry),
    });
    if (res.ok) {
      showToast(`เพิ่ม "${entry.name}" สำเร็จ!`, "success");
      form.reset();
      loadData();
    } else {
      const err = await res.json();
      showToast(err.error || "เกิดข้อผิดพลาด", "error");
    }
  } catch (e) {
    showToast("ไม่สามารถเชื่อมต่อ Server ได้", "error");
  }
}

async function deleteData(id) {
  if (!confirm("ต้องการลบข้อมูลนี้?")) return;
  try {
    await fetch(`${API_BASE}/data/${id}`, { method: "DELETE" });
    showToast("ลบข้อมูลสำเร็จ", "info");
    loadData();
  } catch (e) {
    showToast("เกิดข้อผิดพลาด", "error");
  }
}

async function trainModel() {
  const btn = document.getElementById("train-btn");
  btn.disabled = true;
  btn.innerHTML = `<span class="loading"></span> กำลัง Train...`;

  try {
    const res = await fetch(`${API_BASE}/train`, { method: "POST" });
    const info = await res.json();
    if (res.ok) {
      showToast("Train Model สำเร็จ!", "success");
      renderModelInfo(info);
    } else {
      showToast(info.error || "Train ไม่สำเร็จ", "error");
    }
  } catch (e) {
    showToast("ไม่สามารถเชื่อมต่อ Server ได้", "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `🚀 Train Model`;
  }
}

async function loadModelInfo() {
  try {
    const res = await fetch(`${API_BASE}/model-info`);
    const info = await res.json();
    if (info.trained) {
      renderModelInfo(info);
    } else {
      document.getElementById("model-results").innerHTML = `
        <div class="no-model">
          <div class="no-model-icon">🧠</div>
          <p>ยังไม่ได้ Train Model — กด Train เพื่อเริ่มต้น</p>
        </div>`;
    }
  } catch (e) { }
}

function renderModelInfo(info) {
  const container = document.getElementById("model-results");
  const featureLabels = {
    distance: "ระยะทาง (กม.)", room_size: "ขนาดห้อง (ตร.ม.)",
    convenience: "การเดินทาง", fitness: "ฟิตเนส", room_condition: "สภาพห้อง",
  };

  const coeffRows = Object.entries(info.coefficients).map(([key, val]) => `
    <tr>
      <td>${featureLabels[key] || key}</td>
      <td class="${val >= 0 ? "positive" : "negative"}">${val >= 0 ? "+" : ""}${val.toLocaleString()}</td>
    </tr>`).join("");

  const allValues = [...(info.test_actual || []), ...(info.test_pred || []), ...(info.train_actual || []), ...(info.train_pred || [])];
  const maxVal = Math.max(...allValues, 1);

  let chartHTML = "";
  if (info.test_actual && info.test_actual.length > 0) {
    chartHTML = info.test_actual.map((actual, i) => `
      <div class="chart-bar-group">
        <div class="chart-label">Test ${i + 1}</div>
        <div class="chart-bars">
          <div class="chart-bar actual" style="width: ${(actual / maxVal) * 100}%">
            <span class="chart-bar-value">฿${Number(actual).toLocaleString()}</span>
          </div>
          <div class="chart-bar predicted" style="width: ${(info.test_pred[i] / maxVal) * 100}%">
            <span class="chart-bar-value">฿${Number(info.test_pred[i]).toLocaleString()}</span>
          </div>
        </div>
      </div>`).join("");
  }

  container.innerHTML = `
    <div class="stats-grid">
      <div class="stat-card blue"><div class="stat-value">${info.total_data}</div><div class="stat-label">ข้อมูลทั้งหมด</div></div>
      <div class="stat-card green"><div class="stat-value">${info.train_size}</div><div class="stat-label">Train Set (80%)</div></div>
      <div class="stat-card orange"><div class="stat-value">${info.test_size}</div><div class="stat-label">Test Set (20%)</div></div>
      <div class="stat-card purple"><div class="stat-value">${typeof info.train_r2 === "number" ? (info.train_r2 * 100).toFixed(1) + "%" : info.train_r2}</div><div class="stat-label">Train R²</div></div>
      <div class="stat-card cyan"><div class="stat-value">${typeof info.test_r2 === "number" ? (info.test_r2 * 100).toFixed(1) + "%" : info.test_r2}</div><div class="stat-label">Test R²</div></div>
      <div class="stat-card pink"><div class="stat-value">฿${Number(info.test_mae).toLocaleString()}</div><div class="stat-label">MAE</div></div>
    </div>

    <div class="card">
      <div class="card-title"><span class="icon">📐</span> Regression Equation</div>
      <div class="equation-box">
        Price = ${info.intercept} ${Object.entries(info.coefficients).map(([k, v]) => `${v >= 0 ? "+" : ""}${v} × ${k}`).join(" ")}
      </div>
    </div>

    <div class="card">
      <div class="card-title"><span class="icon">📊</span> Coefficients</div>
      <div class="table-wrapper">
        <table class="coeff-table">
          <thead><tr><th>Feature</th><th>Coefficient</th></tr></thead>
          <tbody>
            ${coeffRows}
            <tr><td><strong>Intercept (ค่าคงที่)</strong></td><td class="${info.intercept >= 0 ? "positive" : "negative"}"><strong>${info.intercept.toLocaleString()}</strong></td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-title"><span class="icon">📈</span> Evaluation Metrics</div>
      <div class="stats-grid">
        <div class="stat-card green"><div class="stat-value">${typeof info.test_r2 === "number" ? info.test_r2.toFixed(4) : info.test_r2}</div><div class="stat-label">R² Score</div></div>
        <div class="stat-card orange"><div class="stat-value">฿${Number(info.test_mae).toLocaleString()}</div><div class="stat-label">MAE</div></div>
        <div class="stat-card pink"><div class="stat-value">฿${Number(info.test_mse).toLocaleString()}</div><div class="stat-label">MSE</div></div>
        <div class="stat-card cyan"><div class="stat-value">฿${Number(info.test_rmse).toLocaleString()}</div><div class="stat-label">RMSE</div></div>
      </div>
    </div>

    ${chartHTML ? `
    <div class="card">
      <div class="card-title"><span class="icon">📉</span> Actual vs Predicted (Test Set)</div>
      <div class="chart-container">${chartHTML}</div>
      <div class="chart-legend">
        <div class="chart-legend-item"><div class="chart-legend-dot actual-dot"></div> Actual</div>
        <div class="chart-legend-item"><div class="chart-legend-dot predicted-dot"></div> Predicted</div>
      </div>
    </div>` : ""}
  `;
}

async function predict(e) {
  e.preventDefault();
  const form = e.target;
  const body = {
    distance: parseFloat(form.pred_distance.value),
    room_size: parseFloat(form.pred_room_size.value),
    convenience: parseInt(form.pred_convenience.value),
    fitness: parseInt(form.pred_fitness.value),
    room_condition: parseInt(form.pred_room_condition.value),
  };

  const btn = document.getElementById("predict-btn");
  btn.disabled = true;
  btn.innerHTML = `<span class="loading"></span> กำลังทำนาย...`;

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const result = await res.json();

    if (res.ok) {
      const resultDiv = document.getElementById("prediction-result");
      const placeholder = document.getElementById("prediction-placeholder");
      resultDiv.classList.remove("show");
      void resultDiv.offsetWidth;
      document.getElementById("predicted-price").textContent = `฿${Number(result.predicted_price).toLocaleString()}`;
      resultDiv.classList.add("show");
      if (placeholder) placeholder.style.display = "none";
      showToast("ทำนายราคาสำเร็จ!", "success");
    } else {
      showToast(result.error || "เกิดข้อผิดพลาด", "error");
    }
  } catch (e) {
    showToast("ไม่สามารถเชื่อมต่อ Server ได้", "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `🔮 ทำนายราคา`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
});
