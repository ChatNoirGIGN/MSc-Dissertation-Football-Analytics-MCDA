# Appendix B: Installation of required libraries

# Run these commands in the terminal to install the libraries:
# pip install streamlit pandas numpy scikit-learn plotly kagglehub scipy

# Finally, the user must run the Streamlit application on localhost, open the terminal (or console)
# in the folder where the Python file (mcdm_tool_2024_2025.py) is located and type this command:
# streamlit run mcdm_tool_2024_2025.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import warnings
import kagglehub
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau

warnings.filterwarnings('ignore')

# ===== CONFIGURACIÓN DE PÁGINA =====
st.set_page_config(
    page_title="Football DSS - Player Selection Optimization",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== ESTILOS CSS MEJORADOS =====
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #AC1E44 0%, #000000 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #AC1E44 0%, #8B1538 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(172, 30, 68, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 500;
    }
    .metric-card h2 {
        margin: 0.5rem 0 0 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .stSelectbox > div > div {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 2px solid #AC1E44;
        border-radius: 10px;
    }
    .stSelectbox > div > div > div {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    .debug-info {
        background-color: #1e1e1e;
        color: #00ff00;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
    }
    .success-msg {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)


# ===== FUNCIONES MCDM =====
def entropy_weights(data):
    data_clean = np.abs(data) + 1e-10
    data_clean = np.nan_to_num(data_clean, nan=1e-10, posinf=1e10, neginf=1e-10)
    data_norm = data_clean / np.sum(data_clean, axis=0)
    data_norm = np.clip(data_norm, 1e-10, 1.0)
    entropy = -np.sum(data_norm * np.log(data_norm), axis=0) / np.log(len(data_norm))
    entropy = np.nan_to_num(entropy, nan=0.5)
    weights = (1 - entropy) / np.sum(1 - entropy)
    return weights / np.sum(weights)


def critic_weights(data):
    data_clean = np.nan_to_num(data, nan=0, posinf=1e10, neginf=-1e10)
    std_dev = np.std(data_clean, axis=0)
    std_dev = np.where(std_dev == 0, 1e-10, std_dev)
    corr_matrix = np.corrcoef(data_clean.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0)
    conflict = np.sum(1 - np.abs(corr_matrix), axis=1)
    critic_values = std_dev * conflict
    weights = critic_values / np.sum(critic_values)
    return weights / np.sum(weights)


def random_forest_weights(data, target_col='Gls'):
    data_clean = data.fillna(0)
    if target_col not in data_clean.columns:
        return np.ones(len(data_clean.columns)) / len(data_clean.columns)
    X = data_clean.drop(columns=[target_col])
    y = data_clean[target_col]
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    weights = np.zeros(len(data_clean.columns))
    feature_indices = [i for i, col in enumerate(data_clean.columns) if col != target_col]
    for i, importance in enumerate(importances):
        weights[feature_indices[i]] = importance
    weights[data_clean.columns.get_loc(target_col)] = np.mean(importances)
    return weights / np.sum(weights)


def topsis_ranking(data, weights, is_beneficial=None):
    # Replace NaN and infinite values
    data_clean = np.nan_to_num(data, nan=0, posinf=1e10, neginf=-1e10)

    # Eliminate constant columns to avoid division by zero
    std_dev = np.std(data_clean, axis=0)
    non_const_cols = std_dev > 1e-10
    data_clean = data_clean[:, non_const_cols]
    weights = weights[non_const_cols]
    if is_beneficial is not None:
        is_beneficial = np.array(is_beneficial)[non_const_cols]
    else:
        is_beneficial = [True] * data_clean.shape[1]

    weights = weights / np.sum(weights)
    norm_data = data_clean / np.sqrt((data_clean ** 2).sum(axis=0))
    weighted_data = norm_data * weights
    ideal_pos = np.where(is_beneficial, weighted_data.max(axis=0), weighted_data.min(axis=0))
    ideal_neg = np.where(is_beneficial, weighted_data.min(axis=0), weighted_data.max(axis=0))
    dist_pos = np.sqrt(((weighted_data - ideal_pos) ** 2).sum(axis=1))
    dist_neg = np.sqrt(((weighted_data - ideal_neg) ** 2).sum(axis=1))
    return dist_neg / (dist_pos + dist_neg)


# ===== MAIN =====
def main():
    ndcg = None
    tau = None
    st.markdown("""
    <div class="main-header">
        <h1>⚽ Football DSS</h1>
        <h3>Decision Support System for Player Selection Optimization</h3>
        <p>Integrating Machine Learning & Multi-Criteria Decision Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        data_source = st.selectbox("📊 Data Source",
                                   ["Kaggle Dataset", "AC Milan Sample Data"])

        # NEW FILTERS
        selected_league, selected_club = "All Leagues", "All Clubs"
        if data_source == "Kaggle Dataset":
            path = kagglehub.dataset_download("hubertsidorowicz/football-players-stats-2024-2025")
            df = pd.read_csv(f'{path}/players_data-2024_2025.csv')
            st.markdown(f'<div class="success-msg">✅ Loaded Kaggle dataset ({len(df)} players)</div>',
                        unsafe_allow_html=True)

            # League Filter
            leagues = ["All Leagues"] + sorted(df['Comp'].dropna().unique().tolist())
            selected_league = st.selectbox("🏟️ Select League", leagues)
            if selected_league != "All Leagues":
                df = df[df['Comp'] == selected_league]

            # Club Filter
            clubs = ["All Clubs"]
            if selected_league != "All Leagues":
                clubs += sorted(df['Squad'].dropna().unique().tolist())
            selected_club = st.selectbox("🏟️ Select Club", clubs)
            if selected_club != "All Clubs":
                df = df[df['Squad'] == selected_club]

        else:  # Sample AC Milan
            sample_data = {
                'Player': ['Mike Maignan', 'Theo Hernández', 'Fikayo Tomori', 'Pierre Kalulu',
                           'Davide Calabria', 'Sandro Tonali', 'Ismaël Bennacer', 'Rafael Leão',
                           'Brahim Díaz', 'Olivier Giroud', 'Ante Rebić'],
                'Pos': ['GK', 'DF', 'DF', 'DF', 'DF', 'MF', 'MF', 'FW', 'MF', 'FW', 'FW'],
                'Age': [28, 26, 25, 24, 27, 23, 26, 24, 24, 37, 30],
                'MP': [38, 35, 31, 20, 29, 34, 32, 34, 28, 31, 25],
                'Min': [3420, 3150, 2790, 1800, 2610, 3060, 2880, 3060, 2520, 2790, 2250],
                'Gls': [0, 3, 2, 1, 2, 8, 3, 11, 6, 11, 4],
                'Ast': [0, 8, 1, 0, 3, 7, 4, 10, 4, 4, 2],
                'xG': [0.0, 2.1, 1.8, 0.8, 1.2, 6.2, 2.1, 8.9, 4.2, 9.8, 3.1],
                'xAG': [0.0, 6.8, 0.9, 0.2, 2.1, 5.8, 3.2, 8.1, 3.8, 2.9, 1.8],
                'PrgC': [12, 89, 45, 23, 67, 156, 134, 178, 98, 67, 45],
                'PrgP': [245, 234, 178, 89, 156, 289, 267, 198, 167, 134, 98]
            }
            df = pd.DataFrame(sample_data)

        # Position filter
        position_options = ["All Positions", "FW", "MF", "DF", "GK"]
        position_filter = st.selectbox("⚽ Player Position",
                                       position_options,
                                       help="Filter players by role")

        weight_method = st.selectbox("🎯 Weight Calculation Method",
                                     ["Random Forest", "Entropy", "CRITIC"])
        min_minutes = st.slider("Minimum Minutes Played", 0, 3000, 500)

    # ==== GLOBAL FILTERS ====
    if 'Min' in df.columns:
        df = df[df['Min'] >= min_minutes]

    if position_filter != "All Positions" and 'Pos' in df.columns:
        def expand_positions(pos):
            if pd.isna(pos):
                return []
            return [p.strip() for p in pos.split(",")]
        df = df[df['Pos'].apply(lambda x: position_filter in expand_positions(x))]

    # ===== KEY METRICS =====
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>👥 Total Players</h3><h2>{len(df)}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3>📅 Average Age</h3><h2>{df['Age'].mean():.1f}</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h3>⚽ Total Goals</h3><h2>{df['Gls'].sum()}</h2></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><h3>🎯 Total Assists</h3><h2>{df['Ast'].sum()}</h2></div>", unsafe_allow_html=True)

    # ===== MCDM =====
    st.markdown("## 🎯 Multi-Criteria Decision Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if position_filter == "FW":
        preferred_cols = ['Gls', 'Ast', 'xG', 'xAG', 'Sh', 'SoT', 'PrgR', 'Min']
    elif position_filter == "MF":
        preferred_cols = ['Ast', 'KP', 'xAG', 'PrgP', 'PrgC', 'Gls', 'Min']
    elif position_filter == "DF":
        preferred_cols = ['Tkl', 'Int', 'Clr', 'Blocks', 'PrgP', 'CrdY', 'CrdR', 'Min']
    elif position_filter == "GK":
        preferred_cols = ['GA90', 'Save%', 'CS%', 'PKsv', 'Min']
    else:
        preferred_cols = ['Gls', 'Ast', 'xG', 'xAG', 'Min', 'PrgC', 'PrgP']

    analysis_cols = [c for c in preferred_cols if c in numeric_cols]
    if len(analysis_cols) < 3:
        analysis_cols = numeric_cols[:min(8, len(numeric_cols))]

    if len(analysis_cols) >= 3:
        analysis_df = df[analysis_cols + ['Player']].copy()
        analysis_data = analysis_df[analysis_cols].fillna(0)

        if weight_method == "Random Forest":
            weights = random_forest_weights(analysis_data)
        elif weight_method == "Entropy":
            weights = entropy_weights(analysis_data.values)
        else:
            weights = critic_weights(analysis_data.values)

        # criteria to be minimized
        minimize_cols = ['CrdY', 'CrdR', 'GA90']
        is_beneficial = [False if col in minimize_cols else True for col in analysis_cols]

        # === Generate TOPSIS scores ===
        topsis_scores = topsis_ranking(analysis_data.values, weights, is_beneficial)

        results_df = analysis_df.copy()
        results_df['TOPSIS_Score'] = pd.Series(topsis_scores, index=analysis_df.index)

        # Replace NaN by 0 before slotting
        results_df['TOPSIS_Score'] = results_df['TOPSIS_Score'].fillna(0)

        results_df['Ranking'] = results_df['TOPSIS_Score'].rank(ascending=False).astype(int)
        results_df = results_df.sort_values('Ranking')

        # Define reference metric for true_relevance by position
        if position_filter in ['FW', 'MF']:
            relevance_col = 'Gls'
        elif position_filter == 'DF':
            # Use a representative defensive metric
            relevance_col = 'Tkl' if 'Tkl' in results_df.columns else None
        elif position_filter == 'GK':
            relevance_col = 'Save%' if 'Save%' in results_df.columns else None
        else:
            relevance_col = 'Gls'  # Default fallback

        if relevance_col and relevance_col in results_df.columns:
            true_relevance = results_df[relevance_col].values
        else:
            st.error(f"Reference metric '{relevance_col}' not found for position {position_filter}.")
            true_relevance = np.zeros(len(results_df))

        pred_scores = results_df['TOPSIS_Score'].values

        if len(true_relevance) < 2 or true_relevance.max() == 0:
            ndcg = 0
            tau = 0
        else:
            true_relevance_norm = true_relevance / true_relevance.max()
            ndcg = ndcg_score([true_relevance_norm], [pred_scores])
            tau, _ = kendalltau(true_relevance, pred_scores)
        with st.expander("🔍 Basic Information"):
            ndcg_display = f"{ndcg:.4f}" if ndcg is not None else "N/A"
            tau_display = f"{tau:.4f}" if tau is not None else "N/A"
            st.markdown(f"""
            <div class="debug-info">
            📊 Dataset Shape: {df.shape}<br>
            📋 Columns: {len(df.columns)}<br>
            🎯 Weight Method: {weight_method}<br>
            📈 Data Source: {data_source}<br>
            🔢 Players after filter: {len(df)}<br>
            📊 NDCG Score: {ndcg_display}<br>
            📊 Kendall's Tau: {tau_display}
            </div>
            """, unsafe_allow_html=True)
        # Add Squad and Comp if they exist in original df
        if 'Squad' in df.columns:
            results_df['Squad'] = df.loc[results_df.index, 'Squad']
        if 'Comp' in df.columns:
            results_df['Comp'] = df.loc[results_df.index, 'Comp']

        # ===== RESULTS =====
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 🏆 Player Rankings")

            display_cols = ['Ranking', 'Player']

            # Show 'Squad' only if no specific club is selected
            if 'Squad' in results_df.columns and selected_club == "All Clubs":
                display_cols.append('Squad')

                # Show 'Comp' only if "All Leagues" is selected
            if 'Comp' in results_df.columns and selected_league == "All Leagues":
                display_cols.append('Comp')
            display_cols += ['TOPSIS_Score'] + analysis_cols

            st.dataframe(results_df[display_cols].head(15),
                         use_container_width=True, hide_index=True)
            # ==== Export results ====
            csv_export = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Rankings (CSV)",
                data=csv_export,
                file_name="player_rankings.csv",
                mime="text/csv"
            )

        with col2:
            st.markdown("### 📊 Feature Weights")
            weights_df = pd.DataFrame({'Feature': analysis_cols,
                                       'Weight': weights[:len(analysis_cols)]}).sort_values('Weight', ascending=False)
            fig_weights = px.bar(weights_df, x='Weight', y='Feature', orientation='h',
                                 color='Weight', color_continuous_scale='Reds')
            st.plotly_chart(fig_weights, use_container_width=True)

        # ===== VISUALIZATIONS =====
        st.markdown("### 📈 Performance Analysis")
        col1, col2 = st.columns(2)
        if 'Gls' in results_df and 'Ast' in results_df:
            fig_scatter = px.scatter(results_df.head(20),
                                     x='Gls', y='Ast',
                                     size='TOPSIS_Score', color='Ranking',
                                     hover_name='Player',
                                     title="Goals vs Assists (sized by TOPSIS Score)",
                                     color_continuous_scale='RdYlBu_r')
            col1.plotly_chart(fig_scatter, use_container_width=True)

        top10 = results_df.head(10)
        fig_top = px.bar(top10, x='Player', y='TOPSIS_Score',
                         color='TOPSIS_Score', color_continuous_scale='Reds')
        fig_top.update_xaxes(tickangle=45)
        col2.plotly_chart(fig_top, use_container_width=True)

        # ===== VIEW COMPLETE DATASET =====
        with st.expander("📋 View Squad Data"):
            st.dataframe(df, use_container_width=True)

    else:
        st.error("❌ Not enough numeric columns for analysis.")


if __name__ == "__main__":
    main()