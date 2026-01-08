"""
Evaluation and visualization functions for model comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, List
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    # Simple tabulate replacement
    def tabulate(table, headers="firstrow", tablefmt="grid"):
        if headers == "firstrow":
            header = table[0]
            data = table[1:]
        else:
            header = headers
            data = table
        # Simple text table
        col_widths = [max(len(str(row[i])) for row in [header] + data) for i in range(len(header))]
        sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        lines = [sep]
        lines.append("|" + "|".join(f" {str(header[i]):<{col_widths[i]}} " for i in range(len(header))) + "|")
        lines.append(sep)
        for row in data:
            lines.append("|" + "|".join(f" {str(row[i]):<{col_widths[i]}} " for i in range(len(row))) + "|")
        lines.append(sep)
        return "\n".join(lines)
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def print_shock_detection_metrics(results: Dict, verbose: bool = True):
    """Print formatted metrics table for shock detection models."""
    if 'all_metrics' not in results:
        print("⚠️  No metrics found in results")
        return
    
    all_metrics = results['all_metrics']
    
    model_names = {
        'lr': 'Logistic Regression',
        'rf': 'Random Forest',
        'svm': 'SVM',
        'gbr': 'Gradient Boosting'
    }
    
    table_data = [["Model", "Features", "AUC", "PR-AUC", "Precision", "Recall", "F1", "Thr", "Cal"]]
    
    for key, metrics in sorted(all_metrics.items()):
        model_code = key.split('_')[0]
        feature_type = 'Hybrid' if 'hybrid' in key else 'Price-Only'
        model_name = model_names.get(model_code, model_code)
        
        table_data.append([
            model_name,
            feature_type,
            f"{metrics['auc']:.3f}",
            f"{metrics.get('pr_auc', 0):.3f}",
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1']:.3f}",
            f"{metrics.get('threshold', 0.5):.2f}",
            "Y" if metrics.get('calibrated', False) else "N"
        ])
    
    print("\n" + "="*80)
    print("📊 Shock Detection Metrics Comparison (All Models)")
    print("="*80)
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    
    # Find best models
    best_auc = 0
    best_auc_model = None
    best_f1 = 0
    best_f1_model = None
    
    for key, metrics in all_metrics.items():
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            best_auc_model = key
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_f1_model = key
    
    if best_auc_model:
        model_code = best_auc_model.split('_')[0]
        feature_type = 'Hybrid' if 'hybrid' in best_auc_model else 'Price-Only'
        print(f"\n🏆 Best Model: {model_names.get(model_code, model_code)} ({feature_type})")
        print(f"   AUC: {best_auc:.3f}")
    
    # Hybrid vs Price-Only improvement
    print("\n📈 Hybrid vs Price-Only Improvement:")
    for model_code in ['lr', 'rf', 'svm', 'gbr']:
        price_key = f'{model_code}_price'
        hybrid_key = f'{model_code}_hybrid'
        if price_key in all_metrics and hybrid_key in all_metrics:
            price_auc = all_metrics[price_key]['auc']
            hybrid_auc = all_metrics[hybrid_key]['auc']
            improvement = hybrid_auc - price_auc
            print(f"   {model_names.get(model_code, model_code)}: {improvement:+.3f} AUC")


def plot_shock_detection_results(results: Dict, save_dir: str = 'artifacts'):
    """Create visualizations for shock detection results."""
    if 'all_metrics' not in results or 'all_test_proba' not in results:
        print("⚠️  Missing data for visualization")
        return
    
    all_metrics = results['all_metrics']
    all_test_proba = results['all_test_proba']
    y_test = results.get('y_test')
    
    if y_test is None:
        print("⚠️  y_test not available")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Shock Detection Model Evaluation', fontsize=16, fontweight='bold')
    
    from sklearn.metrics import roc_curve, confusion_matrix
    
    model_names = {
        'lr': 'Logistic Regression',
        'rf': 'Random Forest',
        'svm': 'SVM',
        'gbr': 'Gradient Boosting'
    }
    colors = {'lr': 'blue', 'rf': 'orange', 'svm': 'red', 'gbr': 'green'}
    
    # ROC Curves
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    for key, proba in all_test_proba.items():
        model_code = key.split('_')[0]
        feature_type = 'Hybrid' if 'hybrid' in key else 'Price'
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = all_metrics[key]['auc']
        linestyle = '-' if 'hybrid' in key else '--'
        ax.plot(fpr, tpr, linestyle=linestyle, color=colors.get(model_code, 'gray'),
                label=f"{model_names.get(model_code, model_code)} ({feature_type}, AUC={auc:.3f})")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - All Models')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # AUC Comparison
    ax = axes[0, 1]
    models = ['lr', 'rf', 'svm', 'gbr']
    price_aucs = [all_metrics.get(f'{m}_price', {}).get('auc', 0) for m in models]
    hybrid_aucs = [all_metrics.get(f'{m}_hybrid', {}).get('auc', 0) for m in models]
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, price_aucs, width, label='Price-Only', color='pink', alpha=0.7)
    ax.bar(x + width/2, hybrid_aucs, width, label='Hybrid', color='gold', alpha=0.7)
    ax.set_xlabel('Model')
    ax.set_ylabel('AUC')
    ax.set_title('AUC Comparison Across Models')
    ax.set_xticks(x)
    ax.set_xticklabels([model_names.get(m, m) for m in models], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # F1 Comparison
    ax = axes[0, 2]
    price_f1s = [all_metrics.get(f'{m}_price', {}).get('f1', 0) for m in models]
    hybrid_f1s = [all_metrics.get(f'{m}_hybrid', {}).get('f1', 0) for m in models]
    ax.bar(x - width/2, price_f1s, width, label='Price-Only', color='pink', alpha=0.7)
    ax.bar(x + width/2, hybrid_f1s, width, label='Hybrid', color='gold', alpha=0.7)
    ax.set_xlabel('Model')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([model_names.get(m, m) for m in models], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Confusion matrices for Gradient Boosting
    if 'gbr_price' in all_test_proba and 'gbr_hybrid' in all_test_proba:
        # Price-only
        ax = axes[1, 0]
        y_pred_price = results['all_test_pred']['gbr_price']
        cm_price = confusion_matrix(y_test, y_pred_price)
        sns.heatmap(cm_price, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Gradient Boosting (Price-Only): Confusion Matrix')
        
        # Hybrid
        ax = axes[1, 1]
        y_pred_hybrid = results['all_test_pred']['gbr_hybrid']
        cm_hybrid = confusion_matrix(y_test, y_pred_hybrid)
        sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Greens', ax=ax, cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Gradient Boosting (Hybrid): Confusion Matrix')
    
    # Predicted probabilities distribution (best model)
    ax = axes[1, 2]
    best_model_key = max(all_metrics.items(), key=lambda x: x[1]['auc'])[0]
    best_proba = all_test_proba[best_model_key]
    ax.hist(best_proba[y_test == 0], bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    ax.hist(best_proba[y_test == 1], bins=50, alpha=0.7, label='Shock', color='red', density=True)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title(f'Predicted Probabilities (Best Model)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to both artifacts and figures
    for save_base_dir in ['artifacts', 'figures']:
        save_path = Path(save_base_dir) / 'shock_detection_results.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved shock detection plots to {save_path}")
    plt.close()


def plot_feature_importance_shock(model, feature_names, top_n=20, save_dir='artifacts/shap'):
    """Plot feature importance as bar chart for shock detection model."""
    print(f"   Generating feature importance plot...")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get feature importance from model
        model_type = type(model).__name__
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models (Random Forest, Gradient Boosting, etc.)
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models (Logistic Regression, etc.)
            # Use absolute coefficients as importance
            importances = np.abs(model.coef_[0])
        else:
            print("⚠️  Model does not support feature importance extraction")
            return
        
        # Ensure feature_names is a list and has correct length
        if feature_names is None or len(feature_names) != len(importances):
            # Generate default feature names if not provided or wrong length
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
            print(f"   ⚠️  Using default feature names (got {len(feature_names) if feature_names else 0}, need {len(importances)})")
        
        # Get top N features
        top_indices = np.argsort(importances)[-top_n:][::-1]  # Sort descending
        top_importances = importances[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        # Create bar chart with color coding by feature type
        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.5)))
        
        # Color code features by type - check interactions FIRST (more specific)
        colors = []
        news_count = 0
        interaction_count = 0
        price_count = 0
        
        for name in top_names:
            # Check for interactions first (most specific pattern)
            if 'interaction' in name.lower():
                colors.append('green')  # Price-news interactions
                interaction_count += 1
            elif 'news' in name.lower() or 'selected' in name.lower():
                colors.append('orange')  # News features
                news_count += 1
            else:
                colors.append('steelblue')  # Price features
                price_count += 1
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(top_names)), top_importances, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for i, (bar, imp) in enumerate(zip(bars, top_importances)):
            ax.text(imp, i, f' {imp:.4f}', va='center', fontsize=9)
        
        # Set labels
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=10)
        ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        
        # Add summary statistics to title
        title_text = f'Top {top_n} Most Important Features for Shock Detection\n'
        title_text += f'Price: {price_count} | News: {news_count} | Interactions: {interaction_count}'
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='steelblue', alpha=0.8, label='Price Features'),
            Patch(facecolor='orange', alpha=0.8, label='News Features'),
            Patch(facecolor='green', alpha=0.8, label='Price-News Interactions')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        # Invert y-axis to show highest importance at top
        ax.invert_yaxis()
        
        # Print analysis
        print(f"\n📊 Feature Importance Analysis:")
        print(f"   Price features in top {top_n}: {price_count} ({price_count/top_n*100:.1f}%)")
        print(f"   News features in top {top_n}: {news_count} ({news_count/top_n*100:.1f}%)")
        print(f"   Interaction features in top {top_n}: {interaction_count} ({interaction_count/top_n*100:.1f}%)")
        
        # Calculate average importance by type
        price_imp = [imp for imp, name in zip(top_importances, top_names) if 'news' not in name.lower() and 'interaction' not in name.lower()]
        news_imp = [imp for imp, name in zip(top_importances, top_names) if 'news' in name.lower() or 'selected' in name.lower()]
        interaction_imp = [imp for imp, name in zip(top_importances, top_names) if 'interaction' in name.lower()]
        
        if price_imp:
            print(f"   Average price feature importance: {np.mean(price_imp):.4f}")
        if news_imp:
            print(f"   Average news feature importance: {np.mean(news_imp):.4f}")
        if interaction_imp:
            print(f"   Average interaction importance: {np.mean(interaction_imp):.4f}")
        
        plt.tight_layout()
        
        # Save to both artifacts and figures
        for save_base_dir in ['artifacts', 'figures']:
            save_path_fig = Path(save_base_dir) / 'shap' / 'feature_importance_shock.png'
            save_path_fig.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path_fig, dpi=150, bbox_inches='tight')
            print(f"💾 Saved feature importance plot to {save_path_fig}")
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Feature importance plot failed: {e}")
        import traceback
        traceback.print_exc()


def compare_models(results):
    """Compare models for regression tasks (single split)."""
    if results is None or results.get('use_walk_forward'):
        return None, None, None, None

    y_test = results.get('y_test')
    if y_test is None:
        return None, None, None, None

    pred_arima = results.get('pred_arima')
    pred_price = results.get('pred_price')
    pred_hybrid = results.get('pred_hybrid')

    def _safe(d):
        return d if isinstance(d, dict) else None

    return _safe(results.get('metrics_arima')), _safe(results.get('metrics_price')), _safe(results.get('metrics_hybrid')), None

def compare_models_walk_forward(results):
    """Compare models using walk-forward validation (aggregate over windows)."""
    if results is None or not results.get('use_walk_forward'):
        return {}, {}

    metrics_by_window = results.get('metrics_by_window', [])
    if not metrics_by_window:
        return {}, {}

    # Build a tidy frame
    rows = []
    for entry in metrics_by_window:
        w = entry.get('window')
        for model_key in ['arima', 'price', 'hybrid']:
            m = entry.get(model_key, {})
            rows.append({
                'window': w,
                'model': model_key,
                'RMSE': m.get('RMSE', np.nan),
                'MAE': m.get('MAE', np.nan),
                'R²': m.get('R²', np.nan),
                'Directional Accuracy': m.get('Directional Accuracy', np.nan),
            })
    window_df = pd.DataFrame(rows)

    aggregated = {}
    for model_key in ['arima', 'price', 'hybrid']:
        sub = window_df[window_df['model'] == model_key]
        aggregated[model_key] = {}
        for metric in ['RMSE', 'MAE', 'R²', 'Directional Accuracy']:
            aggregated[model_key][metric] = {
                'mean': float(sub[metric].mean()),
                'median': float(sub[metric].median()),
            }

    return aggregated, window_df

def plot_forecasts(results, save_path):
    """Plot forecast comparisons for regression tasks."""
    if results is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if results.get('use_walk_forward'):
        # Plot last window only (clean, aligns with report use)
        window_info = results.get('window_info', [])
        if not window_info:
            return
        last = window_info[-1]
        w = last['window']
        preds_arima = results.get('preds_arima', [])
        preds_price = results.get('preds_price', [])
        preds_hybrid = results.get('preds_hybrid', [])
        if w >= len(preds_arima) or w >= len(preds_price) or w >= len(preds_hybrid):
            return

        y_start = last['test_start']
        y_end = last['test_end']
        # We don't store full y, so plot predictions only with index range
        x = np.arange(y_start, y_end)
        plt.figure(figsize=(12, 5))
        plt.plot(x, preds_arima[w], label='ARIMA', alpha=0.9)
        plt.plot(x, preds_price[w], label='ML Baseline (Price)', alpha=0.9)
        plt.plot(x, preds_hybrid[w], label='Hybrid (Price+News)', alpha=0.9)
        plt.title(f'Forecasts (last walk-forward window {w})')
        plt.xlabel('Time index')
        plt.ylabel('Prediction')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return

    y_test = results.get('y_test')
    pred_arima = results.get('pred_arima')
    pred_price = results.get('pred_price')
    pred_hybrid = results.get('pred_hybrid')
    if y_test is None or pred_arima is None or pred_price is None or pred_hybrid is None:
        return

    x = np.arange(len(y_test))
    plt.figure(figsize=(12, 5))
    plt.plot(x, y_test, label='Actual', color='black', linewidth=2, alpha=0.8)
    plt.plot(x, pred_arima, label='ARIMA', alpha=0.9)
    plt.plot(x, pred_price, label='ML Baseline (Price)', alpha=0.9)
    plt.plot(x, pred_hybrid, label='Hybrid (Price+News)', alpha=0.9)
    plt.title('Forecast comparison (test split)')
    plt.xlabel('Test index')
    plt.ylabel('Target')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """Plot feature importance (works for tree models and linear models)."""
    if model is None:
        return
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, 'feature_importances_'):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, 'coef_'):
        coef = np.asarray(model.coef_).reshape(-1)
        importances = np.abs(coef)
    else:
        return

    if not feature_names or len(feature_names) != len(importances):
        feature_names = [f'feature_{i}' for i in range(len(importances))]

    idx = np.argsort(importances)[-top_n:][::-1]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    plt.figure(figsize=(10, max(6, top_n * 0.35)))
    plt.barh(range(len(vals)), vals, color='steelblue', alpha=0.85)
    plt.yticks(range(len(vals)), names)
    plt.gca().invert_yaxis()
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_correlation_heatmap(df, feature_names, save_path=None):
    """Plot correlation heatmap for selected features."""
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or not feature_names:
        return

    cols = [c for c in feature_names if c in df.columns]
    if len(cols) < 2:
        return

    corr = df[cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True, cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_summary_table(results, save_path=None):
    """Create summary metrics table for regression tasks."""
    if results is None:
        return pd.DataFrame()

    rows = []
    if results.get('use_walk_forward'):
        _, window_df = compare_models_walk_forward(results)
        if isinstance(window_df, pd.DataFrame) and len(window_df) > 0:
            # aggregate across windows
            for model_key in ['arima', 'price', 'hybrid']:
                sub = window_df[window_df['model'] == model_key]
                rows.append({
                    'model': model_key,
                    'RMSE_mean': sub['RMSE'].mean(),
                    'RMSE_median': sub['RMSE'].median(),
                    'MAE_mean': sub['MAE'].mean(),
                    'MAE_median': sub['MAE'].median(),
                    'R2_mean': sub['R²'].mean(),
                    'R2_median': sub['R²'].median(),
                    'DA_mean': sub['Directional Accuracy'].mean(),
                    'DA_median': sub['Directional Accuracy'].median(),
                })
            summary = pd.DataFrame(rows)
        else:
            summary = pd.DataFrame()
    else:
        ma = results.get('metrics_arima', {})
        mp = results.get('metrics_price', {})
        mh = results.get('metrics_hybrid', {})
        for model_key, m in [('arima', ma), ('price', mp), ('hybrid', mh)]:
            if isinstance(m, dict) and m:
                rows.append({'model': model_key, **m})
        summary = pd.DataFrame(rows)

    if save_path is not None and len(summary) > 0:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(save_path, index=False)
    return summary


def plot_regression_metrics_table(summary_df: pd.DataFrame, save_path: str, title: str = 'Regression Metrics Summary'):
    """
    Render a compact table (PNG) from a regression summary DataFrame for reports.
    """
    if summary_df is None or len(summary_df) == 0:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = summary_df.copy()
    # Keep a small, report-friendly set of columns
    preferred_cols = [c for c in df.columns if c in [
        'model', 'RMSE', 'MAE', 'R²', 'Directional Accuracy',
        'RMSE_median', 'MAE_median', 'R2_median', 'DA_median',
        'RMSE_mean', 'MAE_mean', 'R2_mean', 'DA_mean',
    ]]
    if preferred_cols:
        df = df[preferred_cols]

    # Round numeric cols for display
    for c in df.columns:
        if c != 'model' and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(float).round(4)

    fig_h = max(3.5, 0.5 + 0.5 * (len(df) + 1))
    fig_w = max(10, 1.2 * len(df.columns))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Style header row
    for j in range(len(df.columns)):
        cell = table[(0, j)]
        cell.set_facecolor('#4CAF50')
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def explain_with_shap(model, X_test, feature_names, top_n=20, save_dir='artifacts/shap'):
    """Generate SHAP explanations for regression models."""
    if not SHAP_AVAILABLE or model is None:
        return
    try:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # TreeExplainer for tree models
        explainer = shap.Explainer(model)
        X_small = X_test[: min(len(X_test), 500)]
        shap_values = explainer(X_small)
        shap.summary_plot(shap_values, feature_names=feature_names, show=False, max_display=top_n)
        plt.tight_layout()
        plt.savefig(save_dir / 'shap_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
    except Exception:
        return

def analyze_prediction_errors(results, feature_names, X_test):
    """Analyze prediction errors."""
    return


def plot_price_with_shocks(df: pd.DataFrame, results: Dict, save_dir: str = 'figures'):
    """
    Plot copper price with shock events highlighted.
    Red markers = all actual shocks
    Green markers = correctly predicted shocks
    """
    if 'price_shock' not in df.columns:
        print("⚠️  price_shock column not found")
        return
    
    # Get test predictions from best hybrid model
    best_model_key = None
    best_auc = 0
    if 'all_metrics' in results:
        for key, metrics in results['all_metrics'].items():
            if 'hybrid' in key and metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                best_model_key = key
    
    if best_model_key is None or 'all_test_pred' not in results:
        print("⚠️  Best hybrid model predictions not available")
        return
    
    y_pred = results['all_test_pred'][best_model_key]
    y_test = results.get('y_test')
    
    if y_test is None:
        print("⚠️  y_test not available")
        return
    
    # Get dates for test set using stored indices
    test_indices = results.get('test_indices')
    if test_indices is not None:
        # Use stored indices from time-based split
        df_test = df.iloc[test_indices].copy()
    else:
        # Fallback: assume test set is the last portion (for backward compatibility)
        test_size = len(y_test)
        df_test = df.iloc[-test_size:].copy()
    
    df_test['predicted_shock'] = y_pred
    df_test['actual_shock'] = df_test['price_shock'].values
    
    # Ensure date column exists
    if 'date' not in df_test.columns and df_test.index.name != 'date':
        # Try to use index if it's datetime
        if isinstance(df_test.index, pd.DatetimeIndex):
            df_test['date'] = df_test.index
        else:
            print("⚠️  Cannot determine dates for visualization")
            return
    
    if 'date' in df_test.columns:
        dates = pd.to_datetime(df_test['date'])
    else:
        dates = df_test.index
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot price
    ax.plot(dates, df_test['price'], 'b-', linewidth=1.5, label='Copper Price', alpha=0.7)
    
    # Plot all actual shocks (red)
    actual_shocks = df_test[df_test['actual_shock'] == 1]
    if len(actual_shocks) > 0:
        actual_dates = dates[df_test['actual_shock'] == 1]
        ax.scatter(actual_dates, actual_shocks['price'], 
                  c='red', marker='v', s=150, alpha=0.7, 
                  label=f'Actual Shocks ({len(actual_shocks)})', zorder=5)
    
    # Plot correctly predicted shocks (green)
    correct_predictions = df_test[(df_test['actual_shock'] == 1) & (df_test['predicted_shock'] == 1)]
    if len(correct_predictions) > 0:
        correct_dates = dates[(df_test['actual_shock'] == 1) & (df_test['predicted_shock'] == 1)]
        ax.scatter(correct_dates, correct_predictions['price'],
                  c='green', marker='^', s=200, alpha=0.9, edgecolors='darkgreen', linewidths=2,
                  label=f'Correctly Predicted ({len(correct_predictions)})', zorder=6)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Copper Price (USD/ton)', fontsize=12)
    ax.set_title(f'Copper Price with Shock Events\nBest Hybrid Model: {best_model_key} (AUC={best_auc:.3f})', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save
    save_path = Path(save_dir) / 'price_with_shocks.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved price with shocks plot to {save_path}")
    plt.close()


def plot_top_news_events(df: pd.DataFrame, results: Dict, save_dir: str = 'figures', top_n: int = 20):
    """
    Plot top significant news events (mines, sanctions, etc.) that correlate with shocks.
    """
    if 'price_shock' not in df.columns:
        print("⚠️  price_shock column not found")
        return
    
    # Load raw news data to get titles
    news_path = Path('src/data/news/copper_news_all_sources.csv')
    if not news_path.exists():
        print(f"⚠️  News file not found: {news_path}")
        return
    
    try:
        news_df = pd.read_csv(news_path)
        news_df['date'] = pd.to_datetime(news_df['date'])
    except Exception as e:
        print(f"⚠️  Failed to load news data: {e}")
        return
    
    # Get shock dates - ensure they are datetime
    shock_mask = df['price_shock'] == 1
    if 'date' in df.columns:
        shock_dates = pd.to_datetime(df.loc[shock_mask, 'date'])
    elif isinstance(df.index, pd.DatetimeIndex):
        shock_dates = df.index[shock_mask]
    else:
        # Try to convert index to datetime
        try:
            shock_dates = pd.to_datetime(df.index[shock_mask])
        except:
            print("⚠️  Cannot determine shock dates")
            return
    
    if len(shock_dates) == 0:
        print("⚠️  No shock dates found")
        return
    
    # Find news within 3 days of shocks - sample from all shocks to get diverse time periods
    significant_news = []
    # Sample shocks evenly across time periods to get diverse news
    n_shocks_to_check = min(100, len(shock_dates))  # Check more shocks for better coverage
    if len(shock_dates) > n_shocks_to_check:
        # Sample evenly across time
        step = len(shock_dates) // n_shocks_to_check
        sampled_shock_dates = shock_dates[::step][:n_shocks_to_check]
    else:
        sampled_shock_dates = shock_dates
    
    for shock_date in sampled_shock_dates:
        if not isinstance(shock_date, pd.Timestamp):
            shock_date = pd.to_datetime(shock_date)
        date_start = shock_date - pd.Timedelta(days=3)
        date_end = shock_date + pd.Timedelta(days=3)
        
        nearby_news = news_df[
            (news_df['date'] >= date_start) & 
            (news_df['date'] <= date_end)
        ]
        
        for _, news_row in nearby_news.iterrows():
            title = str(news_row.get('title', ''))
            text = str(news_row.get('text', ''))
            source = str(news_row.get('source', ''))
            
            # Check for significant keywords
            keywords = [
                'mine', 'mining', 'strike', 'closure', 'shutdown', 'production',
                'sanction', 'export ban', 'import ban', 'embargo', 'trade war',
                'shortage', 'supply', 'disruption', 'strike', 'labor',
                'escondida', 'collahuasi', 'codelco', 'bhp', 'freeport',
                'china', 'chile', 'peru', 'congo', 'zambia'
            ]
            
            combined_text = (title + ' ' + text).lower()
            if any(kw in combined_text for kw in keywords):
                significant_news.append({
                    'date': news_row['date'],
                    'shock_date': shock_date,
                    'title': title[:100],  # Truncate
                    'source': source,
                    'text': text[:200] if pd.notna(text) else ''
                })
    
    if len(significant_news) == 0:
        print("⚠️  No significant news found near shocks")
        return
    
    # Deduplicate news by title (normalized)
    seen_titles = set()
    unique_news = []
    for news_item in significant_news:
        # Normalize title for deduplication
        title_norm = news_item['title'].lower().strip()[:80]  # Use first 80 chars
        if title_norm not in seen_titles:
            seen_titles.add(title_norm)
            unique_news.append(news_item)
    
    # Sort by date (chronological) and take top N evenly distributed
    unique_news = sorted(unique_news, key=lambda x: x['date'])
    # Reduce to fewer news items to avoid overlap
    reduced_top_n = min(top_n, 12)  # Limit to 12 news items max
    # Take evenly distributed samples across time
    if len(unique_news) > reduced_top_n:
        step = len(unique_news) // reduced_top_n
        unique_news = unique_news[::step][:reduced_top_n]
    else:
        unique_news = unique_news[:reduced_top_n]
    
    if len(unique_news) == 0:
        print("⚠️  No unique significant news found after deduplication")
        return
    
    # Create visualization with better layout
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Get dates for price plot
    if 'date' in df.columns:
        price_dates = pd.to_datetime(df['date'])
    elif isinstance(df.index, pd.DatetimeIndex):
        price_dates = df.index
    else:
        price_dates = pd.to_datetime(df.index)
    
    # Plot price timeline on secondary axis
    ax2 = ax.twinx()
    price_line = ax2.plot(price_dates, df['price'], 'b-', linewidth=2, alpha=0.4, label='Copper Price (USD/ton)', zorder=1)
    ax2.set_ylabel('Copper Price (USD/ton)', color='blue', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.grid(False)
    
    # Plot shocks on price axis
    shock_mask = df['price_shock'] == 1
    shock_dates_plot = price_dates[shock_mask]
    shock_prices = df.loc[shock_mask, 'price']
    ax2.scatter(shock_dates_plot, shock_prices, 
               c='red', marker='v', s=80, alpha=0.6, label='Price Shocks', zorder=2, edgecolors='darkred', linewidths=1)
    
    # Simplified news visualization - just markers with minimal text
    price_min = df['price'].min()
    price_max = df['price'].max()
    price_range = price_max - price_min
    
    # Single height level above price line - simpler
    news_y_offset = price_max + price_range * 0.12
    
    # Track categories for legend
    categories_seen = set()
    
    # Simple positioning - just place markers above price line
    for i, news_item in enumerate(unique_news):
        date = news_item['date']
        title = news_item['title']
        source = news_item['source']
        
        # Color based on keywords
        title_lower = title.lower()
        if any(kw in title_lower for kw in ['mine', 'mining', 'strike', 'closure', 'shutdown', 'production']):
            color = 'orange'
            category = 'Mine/Production'
        elif any(kw in title_lower for kw in ['sanction', 'ban', 'embargo', 'trade war', 'export restriction']):
            color = 'red'
            category = 'Sanctions/Trade'
        elif any(kw in title_lower for kw in ['war', 'conflict', 'russia', 'ukraine', 'china']):
            color = 'purple'
            category = 'Geopolitical'
        else:
            color = 'blue'
            category = 'Other'
        
        # Simple marker on price line - just a small star indicator
        price_at_date = df.loc[df['date'] == date, 'price'].values
        if len(price_at_date) > 0:
            ax2.scatter(date, price_at_date[0], c=color, s=150, alpha=0.8, zorder=5, 
                      marker='*', edgecolors='black', linewidths=1.5)
        
        categories_seen.add((category, color))
    
    # Set labels and title
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('')  # No label for left axis (we use right axis for price)
    ax.set_title(f'Significant News Events Near Price Shocks\n({len(unique_news)} unique events, evenly distributed across time)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.2, axis='x')
    ax.set_yticklabels([])  # Hide left Y axis labels
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Copper Price'),
        Patch(facecolor='red', alpha=0.6, label='Price Shocks'),
    ]
    for category, color in sorted(categories_seen):
        legend_elements.append(Patch(facecolor=color, alpha=0.7, label=category))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)
    
    # Add text explanation
    explanation_text = (
        "This chart shows significant news events (mines, sanctions, geopolitical events)\n"
        "that occurred within ±3 days of price shocks. News markers are positioned above\n"
        "the price line to avoid overlap. Events are sampled evenly across time periods."
    )
    ax.text(0.02, 0.98, explanation_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save price chart
    save_path = Path(save_dir) / 'top_news_events.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved price chart with news markers to {save_path}")
    plt.close()
    
    # Create a beautiful table for news events
    fig, ax = plt.subplots(figsize=(16, max(8, len(unique_news) * 0.4)))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['#', 'Date', 'Category', 'Title', 'Source', 'Days from Shock']
    
    for i, news_item in enumerate(unique_news, 1):
        date = news_item['date']
        shock_date = news_item['shock_date']
        title = news_item['title']
        source = news_item['source']
        
        # Determine category
        title_lower = title.lower()
        if any(kw in title_lower for kw in ['mine', 'mining', 'strike', 'closure', 'shutdown', 'production']):
            category = 'Mine/Production'
        elif any(kw in title_lower for kw in ['sanction', 'ban', 'embargo', 'trade war', 'export restriction']):
            category = 'Sanctions/Trade'
        elif any(kw in title_lower for kw in ['war', 'conflict', 'russia', 'ukraine', 'china']):
            category = 'Geopolitical'
        else:
            category = 'Other'
        
        # Calculate days from shock
        days_diff = abs((date - shock_date).days)
        
        # Truncate title if too long
        title_display = title[:80] + '...' if len(title) > 80 else title
        
        table_data.append([
            str(i),
            date.strftime('%Y-%m-%d'),
            category,
            title_display,
            source[:30] + '...' if len(source) > 30 else source,
            f"{days_diff} days"
        ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='left', loc='center',
                    colWidths=[0.03, 0.12, 0.15, 0.5, 0.12, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the table
    # Header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_edgecolor('black')
        table[(0, i)].set_linewidth(1.5)
    
    # Data rows - alternate colors and highlight categories
    for i in range(1, len(table_data) + 1):
        # Alternate row colors
        if i % 2 == 0:
            row_color = '#f0f0f0'
        else:
            row_color = 'white'
        
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(row_color)
            table[(i, j)].set_edgecolor('gray')
            table[(i, j)].set_linewidth(0.5)
        
        # Highlight category column
        category = table_data[i-1][2]
        if category == 'Mine/Production':
            table[(i, 2)].set_facecolor('#FFE5B4')  # Light orange
        elif category == 'Sanctions/Trade':
            table[(i, 2)].set_facecolor('#FFB6C1')  # Light red
        elif category == 'Geopolitical':
            table[(i, 2)].set_facecolor('#E6E6FA')  # Light purple
        else:
            table[(i, 2)].set_facecolor('#E0F2F1')  # Light blue
    
    ax.set_title(f'Significant News Events Near Price Shocks\n({len(unique_news)} unique events)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save table
    table_path = Path(save_dir) / 'top_news_events_table.png'
    table_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(table_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved news events table to {table_path}")
    plt.close()
    
    # Also save as CSV for easy viewing
    csv_path = Path(save_dir) / 'top_news_events.csv'
    news_df_table = pd.DataFrame(table_data, columns=headers)
    news_df_table.to_csv(csv_path, index=False)
    print(f"💾 Saved news events CSV to {csv_path}")


def plot_news_statistics(save_dir: str = 'figures'):
    """
    Create comprehensive news statistics visualizations:
    1. Distribution by source
    2. Distribution over time
    3. Specific news types (mine closures, sanctions, etc.)
    """
    # Load raw news data
    news_path = Path('src/data/news/copper_news_all_sources.csv')
    if not news_path.exists():
        print(f"⚠️  News file not found: {news_path}")
        return
    
    try:
        news_df = pd.read_csv(news_path)
        news_df['date'] = pd.to_datetime(news_df['date'])
    except Exception as e:
        print(f"⚠️  Failed to load news data: {e}")
        return
    
    if len(news_df) == 0:
        print("⚠️  No news data available")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Distribution by source (top 15)
    ax1 = fig.add_subplot(gs[0, 0])
    source_counts = news_df['source'].value_counts().head(15)
    colors_source = plt.cm.Set3(np.linspace(0, 1, len(source_counts)))
    bars1 = ax1.barh(range(len(source_counts)), source_counts.values, color=colors_source, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(source_counts)))
    ax1.set_yticklabels(source_counts.index, fontsize=9)
    ax1.set_xlabel('Number of Articles', fontsize=11, fontweight='bold')
    ax1.set_title('Top 15 News Sources', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, source_counts.values)):
        ax1.text(val, i, f' {val:,}', va='center', fontsize=9)
    
    # 2. Distribution over time (monthly)
    ax2 = fig.add_subplot(gs[0, 1])
    news_df['year_month'] = news_df['date'].dt.to_period('M')
    monthly_counts = news_df['year_month'].value_counts().sort_index()
    monthly_counts.index = monthly_counts.index.astype(str)
    
    ax2.plot(range(len(monthly_counts)), monthly_counts.values, marker='o', linewidth=2, markersize=4, color='steelblue')
    ax2.fill_between(range(len(monthly_counts)), monthly_counts.values, alpha=0.3, color='steelblue')
    ax2.set_xlabel('Time (Monthly)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Articles', fontsize=11, fontweight='bold')
    ax2.set_title('News Articles Distribution Over Time', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    
    # Set x-axis labels (show every 12th month to avoid clutter)
    step = max(1, len(monthly_counts) // 10)
    ax2.set_xticks(range(0, len(monthly_counts), step))
    ax2.set_xticklabels([monthly_counts.index[i] for i in range(0, len(monthly_counts), step)], 
                        rotation=45, ha='right', fontsize=8)
    
    # 3. Specific news types analysis
    ax3 = fig.add_subplot(gs[1, :])
    
    # Categorize news by keywords
    def categorize_news(title, text):
        """Categorize news based on keywords."""
        combined = (str(title) + ' ' + str(text)).lower()
        
        categories = {
            'Mine Closure/Shutdown': ['mine closure', 'mine closed', 'mine shutdown', 'shut down mine', 
                                      'ceases operation', 'suspends operation', 'halts production'],
            'Strike/Labor': ['strike', 'striking', 'labor dispute', 'workers strike', 'union strike', 
                            'walkout', 'industrial action'],
            'Production Cut': ['production cut', 'cuts production', 'reduces production', 
                              'production reduction', 'output cut'],
            'Export Ban/Restriction': ['export ban', 'export restriction', 'bans export', 
                                      'export embargo', 'export limit'],
            'Sanctions': ['sanction', 'sanctions', 'embargo', 'trade restriction', 'trade ban'],
            'Mine Opening/Expansion': ['new mine', 'mine opening', 'opens mine', 'new mine opens',
                                       'mine starts', 'capacity expansion', 'expands capacity'],
            'Production Increase': ['production increase', 'increases production', 'raises production',
                                   'production rise', 'output increase'],
            'War/Conflict': ['war', 'conflict', 'military', 'invasion', 'attack', 'russia', 'ukraine'],
            'Geopolitical': ['geopolitical', 'tension', 'crisis', 'diplomatic', 'trade war', 'tariff'],
            'China Related': ['china', 'chinese', 'china trade', 'china export', 'china import']
        }
        
        for category, keywords in categories.items():
            if any(kw in combined for kw in keywords):
                return category
        return 'Other'
    
    news_df['category'] = news_df.apply(lambda row: categorize_news(row.get('title', ''), row.get('text', '')), axis=1)
    category_counts = news_df['category'].value_counts()
    
    # Create horizontal bar chart
    colors_cat = plt.cm.tab10(np.linspace(0, 1, len(category_counts)))
    bars3 = ax3.barh(range(len(category_counts)), category_counts.values, color=colors_cat, edgecolor='black', linewidth=0.5)
    ax3.set_yticks(range(len(category_counts)))
    ax3.set_yticklabels(category_counts.index, fontsize=10)
    ax3.set_xlabel('Number of Articles', fontsize=11, fontweight='bold')
    ax3.set_title('News Articles by Category (Mine Closures, Sanctions, etc.)', fontsize=12, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()
    
    # Add value labels and percentages
    total_news = len(news_df)
    for i, (bar, val) in enumerate(zip(bars3, category_counts.values)):
        pct = (val / total_news) * 100
        ax3.text(val, i, f' {val:,} ({pct:.1f}%)', va='center', fontsize=9)
    
    # 4. Top specific events (mine closures, strikes, etc.)
    ax4 = fig.add_subplot(gs[2, :])
    
    # Filter for high-impact events
    high_impact_keywords = {
        'Mine Closure': ['mine closure', 'mine closed', 'mine shutdown', 'shut down mine', 'ceases operation'],
        'Strike': ['strike', 'striking', 'workers strike', 'union strike', 'walkout'],
        'Export Ban': ['export ban', 'export restriction', 'bans export', 'export embargo'],
        'Sanctions': ['sanction', 'sanctions', 'embargo', 'trade ban'],
        'Production Cut': ['production cut', 'cuts production', 'reduces production', 'output cut']
    }
    
    high_impact_news = []
    for category, keywords in high_impact_keywords.items():
        mask = news_df.apply(
            lambda row: any(kw in (str(row.get('title', '')) + ' ' + str(row.get('text', ''))).lower() 
                           for kw in keywords), axis=1
        )
        category_news = news_df[mask].copy()
        category_news['impact_category'] = category
        high_impact_news.append(category_news)
    
    if high_impact_news:
        high_impact_df = pd.concat(high_impact_news, ignore_index=True)
        high_impact_df = high_impact_df.sort_values('date')
        
        # Group by year and category
        high_impact_df['year'] = high_impact_df['date'].dt.year
        impact_by_year = high_impact_df.groupby(['year', 'impact_category']).size().unstack(fill_value=0)
        
        # Stacked bar chart
        impact_by_year.plot(kind='bar', stacked=True, ax=ax4, colormap='Set2', edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Number of High-Impact Events', fontsize=11, fontweight='bold')
        ax4.set_title('High-Impact News Events Over Time (Mine Closures, Strikes, Sanctions, etc.)', 
                     fontsize=12, fontweight='bold', pad=10)
        ax4.legend(title='Event Type', fontsize=9, title_fontsize=10, loc='upper left')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    
    plt.suptitle('Comprehensive News Statistics Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    save_path = Path(save_dir) / 'news_statistics.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved news statistics to {save_path}")
    plt.close()
    
    # Print summary statistics
    print(f"\n📊 News Statistics Summary:")
    print(f"   Total articles: {len(news_df):,}")
    print(f"   Date range: {news_df['date'].min().strftime('%Y-%m-%d')} to {news_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Unique sources: {news_df['source'].nunique()}")
    print(f"   Top 3 sources: {', '.join(source_counts.head(3).index.tolist())}")
    print(f"   High-impact events: {len(high_impact_df) if high_impact_news else 0}")
