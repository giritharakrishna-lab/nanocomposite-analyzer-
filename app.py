import io

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)

MATRICES = {
    "Epoxy": {
        "name": "Epoxy Resin",
        "E_m": 3.2,
        "k_m": 0.20,
        "sigma_m": 1e-12,
    },
    "PEEK": {
        "name": "PEEK (Biomedical Polymer)",
        "E_m": 3.6,
        "k_m": 0.25,
        "sigma_m": 1e-13,
    },
    "PLA": {
        "name": "PLA (Biodegradable Polymer)",
        "E_m": 2.8,
        "k_m": 0.13,
        "sigma_m": 1e-14,
    },
    "Al6061": {
        "name": "Aluminium 6061",
        "E_m": 68.9,
        "k_m": 167.0,
        "sigma_m": 2.5e7,
    },
}

DEMO_DATA = [
    {"volume_fraction_pct": 0.5, "filler_type": "Graphene", "exp_modulus_gpa": 3.85, "exp_thermal_wmk": 0.34},
    {"volume_fraction_pct": 1.0, "filler_type": "Graphene", "exp_modulus_gpa": 4.45, "exp_thermal_wmk": 0.52},
    {"volume_fraction_pct": 2.0, "filler_type": "Graphene", "exp_modulus_gpa": 5.60, "exp_thermal_wmk": 0.89},
    {"volume_fraction_pct": 3.5, "filler_type": "Graphene", "exp_modulus_gpa": 7.15, "exp_thermal_wmk": 1.42},
    {"volume_fraction_pct": 5.0, "filler_type": "Graphene", "exp_modulus_gpa": 8.65, "exp_thermal_wmk": 1.98},
    {"volume_fraction_pct": 0.5, "filler_type": "CNT", "exp_modulus_gpa": 3.92, "exp_thermal_wmk": 0.31},
    {"volume_fraction_pct": 1.0, "filler_type": "CNT", "exp_modulus_gpa": 4.65, "exp_thermal_wmk": 0.45},
    {"volume_fraction_pct": 2.0, "filler_type": "CNT", "exp_modulus_gpa": 6.10, "exp_thermal_wmk": 0.72},
    {"volume_fraction_pct": 3.5, "filler_type": "CNT", "exp_modulus_gpa": 8.20, "exp_thermal_wmk": 1.15},
    {"volume_fraction_pct": 5.0, "filler_type": "CNT", "exp_modulus_gpa": 10.15, "exp_thermal_wmk": 1.58},
]

REQUIRED_CSV_COLUMNS = [
    "volume_fraction_pct",
    "filler_type",
    "exp_modulus_gpa",
    "exp_thermal_wmk",
]


def compute_properties(matrix_key, vf_array, alpha_gnp, alpha_cnt, orientation, eta_i):
    matrix = MATRICES.get(matrix_key, MATRICES["Epoxy"])
    E_m, k_m, sigma_m = matrix["E_m"], matrix["k_m"], matrix["sigma_m"]

    E_gnp, k_gnp, sigma_gnp = 1000.0, 3000.0, 1e4
    E_cnt, k_cnt, sigma_cnt = 1050.0, 3500.0, 1e4
    E_gnp_eff = E_gnp * eta_i
    E_cnt_eff = E_cnt * eta_i

    xi_gnp = (2.0 / 3.0) * alpha_gnp
    xi_cnt = 2.0 * alpha_cnt

    eta_L_gnp = ((E_gnp_eff / E_m) - 1.0) / ((E_gnp_eff / E_m) + xi_gnp)
    eta_T_gnp = ((E_gnp_eff / E_m) - 1.0) / ((E_gnp_eff / E_m) + 2.0)
    eta_L_cnt = ((E_cnt_eff / E_m) - 1.0) / ((E_cnt_eff / E_m) + xi_cnt)
    eta_T_cnt = ((E_cnt_eff / E_m) - 1.0) / ((E_cnt_eff / E_m) + 2.0)

    E_L_gnp = E_m * (1.0 + xi_gnp * eta_L_gnp * vf_array) / (1.0 - eta_L_gnp * vf_array)
    E_T_gnp = E_m * (1.0 + 2.0 * eta_T_gnp * vf_array) / (1.0 - eta_T_gnp * vf_array)
    E_L_cnt = E_m * (1.0 + xi_cnt * eta_L_cnt * vf_array) / (1.0 - eta_L_cnt * vf_array)
    E_T_cnt = E_m * (1.0 + 2.0 * eta_T_cnt * vf_array) / (1.0 - eta_T_cnt * vf_array)

    if orientation == "aligned":
        E_out_gnp, E_out_cnt = E_L_gnp, E_L_cnt
    else:
        E_out_gnp = 0.49 * E_L_gnp + 0.51 * E_T_gnp
        E_out_cnt = (3.0 / 8.0) * E_L_cnt + (5.0 / 8.0) * E_T_cnt

    R_k = 2.5e-8 / max(eta_i, 0.2)
    L_gnp = alpha_gnp * 5e-9
    L_cnt = alpha_cnt * 10e-9
    k_gnp_eff = k_gnp / (1.0 + (2.0 * R_k * k_gnp) / L_gnp)
    k_cnt_eff = k_cnt / (1.0 + (2.0 * R_k * k_cnt) / L_cnt)
    p_gnp = 1.0 if orientation == "aligned" else (2.0 / 3.0)
    p_cnt = 1.0 if orientation == "aligned" else (1.0 / 3.0)
    k_out_gnp = k_m * (1.0 + p_gnp * vf_array * max((k_gnp_eff - k_m) / k_m, 0.0)) / (1.0 - vf_array)
    k_out_cnt = k_m * (1.0 + p_cnt * vf_array * max((k_cnt_eff - k_m) / k_m, 0.0)) / (1.0 - vf_array)

    phi_c_gnp = 1.5 / alpha_gnp
    phi_c_cnt = 0.7 / alpha_cnt
    t_exp = 2.0
    sigma_out_gnp = np.where(
        vf_array > phi_c_gnp,
        sigma_m + sigma_gnp * eta_i * np.power(np.maximum(vf_array - phi_c_gnp, 0.0), t_exp),
        sigma_m * (1.0 + 2.0 * vf_array),
    )
    sigma_out_cnt = np.where(
        vf_array > phi_c_cnt,
        sigma_m + sigma_cnt * eta_i * np.power(np.maximum(vf_array - phi_c_cnt, 0.0), t_exp),
        sigma_m * (1.0 + 2.0 * vf_array),
    )

    return {
        "E_gnp": E_out_gnp,
        "E_cnt": E_out_cnt,
        "k_gnp": k_out_gnp,
        "k_cnt": k_out_cnt,
        "sigma_gnp": sigma_out_gnp,
        "sigma_cnt": sigma_out_cnt,
        "phi_c_gnp_pct": round(phi_c_gnp * 100.0, 3),
        "phi_c_cnt_pct": round(phi_c_cnt * 100.0, 3),
        "matrix": matrix,
    }


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/simulate", methods=["POST"])
def simulate():
    data = request.get_json()
    matrix_key = data.get("matrix", "Epoxy")
    active_vf_pct = float(data.get("vf_pct", 2.0))
    alpha_gnp = float(data.get("alpha_gnp", 150))
    alpha_cnt = float(data.get("alpha_cnt", 250))
    orientation = data.get("orientation", "random")
    eta_i = float(data.get("eta_i", 0.75))

    vf_pct_grid = np.round(np.linspace(0.0, 10.0, 101), 2)
    vf_grid = vf_pct_grid / 100.0
    result = compute_properties(matrix_key, vf_grid, alpha_gnp, alpha_cnt, orientation, eta_i)

    idx = int(round(active_vf_pct * 10))
    idx = max(0, min(idx, 100))
    thermal_gain_gnp = ((result["k_gnp"][idx] - result["matrix"]["k_m"]) / result["matrix"]["k_m"]) * 100.0
    percolated = (active_vf_pct >= result["phi_c_cnt_pct"]) or (active_vf_pct >= result["phi_c_gnp_pct"])

    return jsonify({
        "vf_pct": vf_pct_grid.tolist(),
        "E_gnp": np.round(result["E_gnp"], 3).tolist(),
        "E_cnt": np.round(result["E_cnt"], 3).tolist(),
        "k_gnp": np.round(result["k_gnp"], 3).tolist(),
        "k_cnt": np.round(result["k_cnt"], 3).tolist(),
        "log_sigma_gnp": np.round(np.log10(result["sigma_gnp"]), 3).tolist(),
        "log_sigma_cnt": np.round(np.log10(result["sigma_cnt"]), 3).tolist(),
        "phi_c_gnp_pct": result["phi_c_gnp_pct"],
        "phi_c_cnt_pct": result["phi_c_cnt_pct"],
        "matrix": result["matrix"],
        "kpi": {
            "active_vf_pct": active_vf_pct,
            "E_gnp": round(float(result["E_gnp"][idx]), 2),
            "E_cnt": round(float(result["E_cnt"][idx]), 2),
            "thermal_gain_pct": round(float(thermal_gain_gnp), 1),
            "percolation_status": "Conductive" if percolated else "Insulating",
        },
    })


@app.route("/api/demo-data", methods=["GET"])
def get_demo_data():
    return jsonify({"source": "Built-in Demo Literature Dataset (Epoxy Matrix)", "rows": DEMO_DATA})


@app.route("/api/sample-csv", methods=["GET"])
def download_sample_csv():
    df = pd.DataFrame(DEMO_DATA)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return Response(
        csv_buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=demo_nanocomposite_input.csv"},
    )


@app.route("/api/upload-csv", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Please upload a .csv file."}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Invalid file type. Only .csv files are accepted."}), 400

    try:
        df = pd.read_csv(file)
        df.columns = [column.strip().lower() for column in df.columns]
        missing = [column for column in REQUIRED_CSV_COLUMNS if column not in df.columns]
        if missing:
            return jsonify({
                "error": f"Missing required column(s): {', '.join(missing)}. Expected: {', '.join(REQUIRED_CSV_COLUMNS)}"
            }), 400

        df["filler_type"] = df["filler_type"].astype(str).str.strip()
        rows = df[REQUIRED_CSV_COLUMNS].dropna().to_dict(orient="records")
        if not rows:
            return jsonify({"error": "The CSV file contains no valid numeric rows."}), 400

        return jsonify({"source": f"Uploaded File: {file.filename}", "rows": rows})
    except Exception as exc:
        return jsonify({"error": f"Failed to parse CSV: {str(exc)}"}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)