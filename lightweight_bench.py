import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from indcpa import make_indcpa_dataset, train_test_split, systems_to_test
from systems import generate_keys

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


WEAK_SYSTEMS = [
    'No Encryption', 'Constant XOR', 'Toy Fixed XOR',
    'Toy Substitution', 'Toy Permutation', 'Toy 1-Round Feistel',
    'AES CTR Fixed Nonce', 'Toy Caesar', 'Toy Repeating XOR',
    'Toy Byte Rotate', 'Toy Mask HighNibble', 'Toy LFSR Stream',
    'Toy 2-Round Feistel',
]

SEMI_WEAK_SYSTEMS = [
    'Semi Reduced Feistel', 'Semi Partial Mask', 'Semi Truncated AES',
    'Semi Nonce Mix', 'Semi LFSR Long', 'Semi Key Rotation',
    'DES NonDet', 'AES CTR Reduced',
]

STRONG_SYSTEMS = [
    'DES', 'AES ECB', 'AES CTR', 'RSA Plain', 'RSA OAEP', 'RSA OAEP Reused','One-Time Pad'
]

MODELS = {
    'logistic_l2': LogisticRegression(max_iter=1000),
    'logistic_l1': LogisticRegression(penalty='l1', solver='saga', max_iter=1000),
    'linear_svc': LinearSVC(max_iter=2000),
    'dtree': DecisionTreeClassifier(max_depth=6),
    'rf_small': RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1),
    'gb_small': GradientBoostingClassifier(n_estimators=100, max_depth=3),
    'knn3': KNeighborsClassifier(n_neighbors=3),
    'gaussian_nb': GaussianNB()
}


def flatten_X_from_make(X):
    return X.reshape(X.shape[0], -1)


def sizeof(obj) -> int:
    return len(pickle.dumps(obj))


def run_bench(cipher_name='Toy Substitution', n_samples=5000, msg_len=16):
    print(f"\nRunning benchmark for {cipher_name} with {n_samples} samples per class")
    keys = generate_keys()
    X, y = make_indcpa_dataset(n_samples=n_samples, msg_len=msg_len, keys=keys, cipher_name=cipher_name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    Xtr = flatten_X_from_make(X_train)
    Xte = flatten_X_from_make(X_test)

    results = []
    for name, model in MODELS.items():
        m = model
        start = time.time()
        try:
            m.fit(Xtr, y_train)
        except Exception as e:
            print(f"Model {name} failed to fit: {e}")
            continue
        train_time = time.time() - start

        start = time.time()
        preds = m.predict(Xte)
        inf_time = time.time() - start
        acc = accuracy_score(y_test, preds)

        size = sizeof(m)

        print(f"{name:12s} | acc={acc:.4f} | train_s={train_time:.3f} | inf_s={inf_time:.3f} | size_bytes={size}")
        results.append((name, acc, train_time, inf_time, size))

    return results


def categorize_results(all_results):
    """Organize results by security strength category."""
    weak_results = {k: v for k, v in all_results.items() if k in WEAK_SYSTEMS}
    semi_results = {k: v for k, v in all_results.items() if k in SEMI_WEAK_SYSTEMS}
    strong_results = {k: v for k, v in all_results.items() if k in STRONG_SYSTEMS}
    return weak_results, semi_results, strong_results


def plot_accuracy_by_category(all_results):
    """Plot accuracy comparison separated by security category."""
    weak, semi, strong = categorize_results(all_results)
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    model_names = list(MODELS.keys())
    
    categories = [
        (weak, 'WEAK Systems (Easily Breakable)', axes[0], 'Reds'),
        (semi, 'SEMI-WEAK Systems (Partial Vulnerabilities)', axes[1], 'Oranges'),
        (strong, 'STRONG Systems (Cryptographically Secure)', axes[2], 'Greens')
    ]
    
    for results, title, ax, cmap in categories:
        if not results:
            continue
            
        ciphers = list(results.keys())
        x = np.arange(len(model_names))
        width = 0.8 / max(len(ciphers), 1)
        
        colors = plt.cm.get_cmap(cmap)(np.linspace(0.3, 0.9, len(ciphers)))
        
        for i, cipher in enumerate(ciphers):
            accuracies = [results[cipher][model][0] for model in model_names]
            offset = width * (i - len(ciphers)/2 + 0.5)
            ax.bar(x + offset, accuracies, width, label=cipher, alpha=0.85, color=colors[i])
        
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(title='Cipher', ncol=2, fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    return fig


def plot_accuracy_diff_matrix(all_results):
    """Create difference matrix comparing model performance across all systems."""
    model_names = list(MODELS.keys())
    system_names = list(all_results.keys())
    
    n_models = len(model_names)
    n_systems = len(system_names)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    weak, semi, strong = categorize_results(all_results)
    categories = [
        (weak, 'Accuracy Differences - WEAK Systems', axes[0]),
        (semi, 'Accuracy Differences - SEMI-WEAK Systems', axes[1]),
        (strong, 'Accuracy Differences - STRONG Systems', axes[2])
    ]
    
    for results, title, ax in categories:
        if not results:
            ax.set_visible(False)
            continue
        
        systems = list(results.keys())
        
        avg_diff_matrix = np.zeros((n_models, n_models))
        
        for system in systems:
            for i, model_i in enumerate(model_names):
                for j, model_j in enumerate(model_names):
                    acc_i = results[system][model_i][0]
                    acc_j = results[system][model_j][0]
                    avg_diff_matrix[i, j] += (acc_i - acc_j)
        
        avg_diff_matrix /= len(systems)
        
        im = ax.imshow(avg_diff_matrix, cmap='RdYlGn', aspect='auto', 
                      vmin=-0.3, vmax=0.3)
        ax.set_xticks(np.arange(n_models))
        ax.set_yticks(np.arange(n_models))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_yticklabels(model_names)
        ax.set_title(title, fontweight='bold', fontsize=12, pad=10)
        ax.set_xlabel('Model (column)', fontweight='bold')
        ax.set_ylabel('Model (row)', fontweight='bold')
        
        for i in range(n_models):
            for j in range(n_models):
                color = 'white' if abs(avg_diff_matrix[i, j]) > 0.15 else 'black'
                text = ax.text(j, i, f'{avg_diff_matrix[i, j]:.3f}',
                             ha="center", va="center", color=color, fontsize=9)
        
        cbar = plt.colorbar(im, ax=ax, label='Avg Accuracy Difference (row - col)')
        cbar.set_label('Avg Accuracy Difference (row - col)', fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_category_performance_heatmap(all_results):
    """Heatmap showing average performance across categories."""
    weak, semi, strong = categorize_results(all_results)
    model_names = list(MODELS.keys())
    
    categories = ['Weak', 'Semi-Weak', 'Strong']
    acc_matrix = np.zeros((len(categories), len(model_names)))
    
    for cat_idx, results in enumerate([weak, semi, strong]):
        if results:
            for model_idx, model in enumerate(model_names):
                accs = [results[sys][model][0] for sys in results.keys()]
                acc_matrix[cat_idx, model_idx] = np.mean(accs)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_yticklabels(categories)
    ax.set_title('Average Model Accuracy by Security Category', fontweight='bold', fontsize=14, pad=15)
    ax.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax.set_ylabel('System Category', fontweight='bold', fontsize=12)
    
    for i in range(len(categories)):
        for j in range(len(model_names)):
            text = ax.text(j, i, f'{acc_matrix[i, j]:.3f}',
                         ha="center", va="center", 
                         color='white' if acc_matrix[i, j] < 0.6 else 'black',
                         fontsize=10, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, label='Average Accuracy')
    cbar.set_label('Average Accuracy', fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_model_effectiveness_across_categories(all_results):
    """Show how each model's effectiveness changes across security categories."""
    weak, semi, strong = categorize_results(all_results)
    model_names = list(MODELS.keys())
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(3)  # Three categories
    width = 0.8 / len(model_names)
    
    colors = plt.cm.tab10(np.arange(len(model_names)))
    
    for i, model in enumerate(model_names):
        weak_acc = np.mean([weak[sys][model][0] for sys in weak.keys()]) if weak else 0
        semi_acc = np.mean([semi[sys][model][0] for sys in semi.keys()]) if semi else 0
        strong_acc = np.mean([strong[sys][model][0] for sys in strong.keys()]) if strong else 0
        
        accs = [weak_acc, semi_acc, strong_acc]
        offset = width * (i - len(model_names)/2 + 0.5)
        ax.bar(x + offset, accs, width, label=model, alpha=0.85, color=colors[i])
    
    ax.set_xlabel('Security Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Effectiveness Across Security Categories', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(['Weak Systems', 'Semi-Weak Systems', 'Strong Systems'])
    ax.legend(title='Model', ncol=2, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Random Guess')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    return fig


def plot_timing_analysis(all_results):
    """Comprehensive timing analysis: training time, inference time, and efficiency metrics."""
    weak, semi, strong = categorize_results(all_results)
    model_names = list(MODELS.keys())
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    categories_data = [weak, semi, strong]
    cat_names = ['Weak', 'Semi-Weak', 'Strong']
    colors_cat = ['#ff6b6b', '#ffa500', '#4caf50']
    
    x = np.arange(len(model_names))
    width = 0.25
    
    for idx, (results, cat_name, color) in enumerate(zip(categories_data, cat_names, colors_cat)):
        if not results:
            continue
        avg_times = []
        for model in model_names:
            times = [results[sys][model][1] for sys in results.keys()]
            avg_times.append(np.mean(times))
        
        ax1.bar(x + idx*width, avg_times, width, label=cat_name, alpha=0.8, color=color)
    
    ax1.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Avg Training Time (s)', fontsize=10, fontweight='bold')
    ax1.set_title('Training Time by Category', fontsize=11, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    ax1.legend(fontsize=9)
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    
    for idx, (results, cat_name, color) in enumerate(zip(categories_data, cat_names, colors_cat)):
        if not results:
            continue
        avg_times = []
        for model in model_names:
            times = [results[sys][model][2] for sys in results.keys()]
            avg_times.append(np.mean(times))
        
        ax2.bar(x + idx*width, avg_times, width, label=cat_name, alpha=0.8, color=color)
    
    ax2.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Avg Inference Time (s)', fontsize=10, fontweight='bold')
    ax2.set_title('Inference Time by Category', fontsize=11, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    ax2.legend(fontsize=9)
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    
    ratios = []
    for model in model_names:
        train_times = [all_results[sys][model][1] for sys in all_results.keys()]
        inf_times = [all_results[sys][model][2] for sys in all_results.keys()]
        ratio = np.mean(train_times) / np.mean(inf_times)
        ratios.append(ratio)
    
    ax3.bar(model_names, ratios, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Train/Inference Ratio', fontsize=10, fontweight='bold')
    ax3.set_title('Training vs Inference Time Ratio', fontsize=11, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    ax3.set_yscale('log')
    ax3.grid(axis='y', alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 0])
    
    efficiency_weak = []
    efficiency_strong = []
    
    for model in model_names:
        if weak:
            accs = [weak[sys][model][0] for sys in weak.keys()]
            times = [weak[sys][model][1] for sys in weak.keys()]
            efficiency_weak.append(np.mean(accs) / np.mean(times))
        
        if strong:
            accs = [strong[sys][model][0] for sys in strong.keys()]
            times = [strong[sys][model][1] for sys in strong.keys()]
            efficiency_strong.append(np.mean(accs) / np.mean(times))
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    if efficiency_weak:
        ax4.bar(x_pos - width/2, efficiency_weak, width, label='Weak Systems', 
               alpha=0.8, color='#ff6b6b')
    if efficiency_strong:
        ax4.bar(x_pos + width/2, efficiency_strong, width, label='Strong Systems', 
               alpha=0.8, color='#4caf50')
    
    ax4.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Accuracy / Train Time', fontsize=10, fontweight='bold')
    ax4.set_title('Training Efficiency (Acc/sec)', fontsize=11, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    ax4.legend(fontsize=9)
    ax4.set_yscale('log')
    ax4.grid(axis='y', alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    
    train_stds = []
    for model in model_names:
        times = [all_results[sys][model][1] for sys in all_results.keys()]
        train_stds.append(np.std(times) / np.mean(times))  
    ax5.bar(model_names, train_stds, alpha=0.7, color='coral', edgecolor='black')
    ax5.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax5.set_ylabel('CV (Std/Mean)', fontsize=10, fontweight='bold')
    ax5.set_title('Training Time Consistency', fontsize=11, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    ax5.grid(axis='y', alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 2])
    
    colors_scatter = plt.cm.tab10(np.arange(len(model_names)))
    
    for idx, model in enumerate(model_names):
        accs = [all_results[sys][model][0] for sys in all_results.keys()]
        times = [all_results[sys][model][1] for sys in all_results.keys()]
        ax6.scatter(times, accs, alpha=0.6, s=30, color=colors_scatter[idx], label=model)
    
    ax6.set_xlabel('Training Time (s)', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
    ax6.set_title('Accuracy vs Training Time', fontsize=11, fontweight='bold')
    ax6.set_xscale('log')
    ax6.legend(fontsize=7, ncol=2)
    ax6.grid(alpha=0.3)
    ax6.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    
    ax7 = fig.add_subplot(gs[2, 0])
    
    time_matrix = np.zeros((len(cat_names), len(model_names)))
    
    for cat_idx, results in enumerate(categories_data):
        if results:
            for model_idx, model in enumerate(model_names):
                times = [results[sys][model][1] for sys in results.keys()]
                time_matrix[cat_idx, model_idx] = np.mean(times)
    
    im = ax7.imshow(time_matrix, cmap='YlOrRd', aspect='auto')
    ax7.set_xticks(np.arange(len(model_names)))
    ax7.set_yticks(np.arange(len(cat_names)))
    ax7.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    ax7.set_yticklabels(cat_names, fontsize=9)
    ax7.set_title('Avg Training Time Heatmap', fontsize=11, fontweight='bold')
    
    for i in range(len(cat_names)):
        for j in range(len(model_names)):
            if time_matrix[i, j] > 0:
                text = ax7.text(j, i, f'{time_matrix[i, j]:.2f}',
                              ha="center", va="center", color='black', fontsize=7)
    
    plt.colorbar(im, ax=ax7, label='Time (s)')
    
    ax8 = fig.add_subplot(gs[2, 1])
    
    speedups = []
    for model in model_names:
        train_times = [all_results[sys][model][1] for sys in all_results.keys()]
        inf_times = [all_results[sys][model][2] for sys in all_results.keys()]
        speedup = np.mean(train_times) / np.mean(inf_times)
        speedups.append(speedup)
    
    colors_speedup = ['green' if s > 100 else 'orange' if s > 10 else 'red' for s in speedups]
    bars = ax8.bar(model_names, speedups, alpha=0.7, color=colors_speedup, edgecolor='black')
    ax8.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax8.set_ylabel('Speedup Factor', fontsize=10, fontweight='bold')
    ax8.set_title('Inference Speedup over Training', fontsize=11, fontweight='bold')
    ax8.tick_params(axis='x', rotation=45)
    plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    ax8.set_yscale('log')
    ax8.grid(axis='y', alpha=0.3)
    
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.0f}x', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax9 = fig.add_subplot(gs[2, 2])
    
    for model in model_names:
        avg_acc = np.mean([all_results[sys][model][0] for sys in all_results.keys()])
        avg_time = np.mean([all_results[sys][model][1] for sys in all_results.keys()])
        avg_size = np.mean([all_results[sys][model][3] for sys in all_results.keys()]) / 1024
        
        ax9.scatter(avg_time, avg_acc, s=avg_size/10, alpha=0.6, 
                   label=model if avg_size/10 > 20 else '')
        ax9.annotate(model, (avg_time, avg_acc), fontsize=7, alpha=0.7)
    
    ax9.set_xlabel('Avg Training Time (s)', fontsize=10, fontweight='bold')
    ax9.set_ylabel('Avg Accuracy', fontsize=10, fontweight='bold')
    ax9.set_title('Pareto Frontier (size = bubble size)', fontsize=11, fontweight='bold')
    ax9.set_xscale('log')
    ax9.grid(alpha=0.3)
    ax9.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    
    return fig


def plot_differential_analysis(all_results):
    """Comprehensive differential analysis showing performance gaps."""
    weak, semi, strong = categorize_results(all_results)
    model_names = list(MODELS.keys())
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :2])
    
    for cat_name, results, color in [('Weak', weak, 'red'), 
                                      ('Semi-Weak', semi, 'orange'), 
                                      ('Strong', strong, 'green')]:
        if not results:
            continue
        all_accs = []
        for system in results.keys():
            for model in model_names:
                all_accs.append(results[system][model][0])
        
        ax1.hist(all_accs, bins=20, alpha=0.5, label=f'{cat_name} (n={len(all_accs)})', 
                color=color, edgecolor='black')
    
    ax1.set_xlabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Accuracy Distribution by Security Category', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 2])
    
    best_models = []
    system_categories = []
    
    for system, results in all_results.items():
        best_acc = 0
        best_model = ''
        for model in model_names:
            if results[model][0] > best_acc:
                best_acc = results[model][0]
                best_model = model
        best_models.append(best_model)
        
        if system in WEAK_SYSTEMS:
            system_categories.append('W')
        elif system in SEMI_WEAK_SYSTEMS:
            system_categories.append('S')
        else:
            system_categories.append('ST')
    
    from collections import Counter
    model_counts = Counter(best_models)
    
    ax2.bar(range(len(model_counts)), list(model_counts.values()), 
           tick_label=list(model_counts.keys()), alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax2.set_ylabel('# Times Best', fontsize=10, fontweight='bold')
    ax2.set_title('Best Performing Model Count', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 0])
    
    categories_data = [weak, semi, strong]
    cat_names = ['Weak', 'Semi-Weak', 'Strong']
    success_rates = []
    
    for results in categories_data:
        if not results:
            success_rates.append(0)
            continue
        total = 0
        success = 0
        for system in results.keys():
            for model in model_names:
                acc = results[system][model][0]
                total += 1
                if acc > 0.6:
                    success += 1
        success_rates.append(success / total if total > 0 else 0)
    
    colors_bar = ['#ff6b6b', '#ffa500', '#4caf50']
    bars = ax3.bar(cat_names, success_rates, alpha=0.7, color=colors_bar, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Success Rate', fontsize=11, fontweight='bold')
    ax3.set_title('Attack Success Rate (Acc > 60%)', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax4 = fig.add_subplot(gs[1, 1])
    
    model_stds = []
    for model in model_names:
        all_accs = [all_results[sys][model][0] for sys in all_results.keys()]
        model_stds.append(np.std(all_accs))
    
    ax4.bar(model_names, model_stds, alpha=0.7, color='coral', edgecolor='black')
    ax4.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Std Dev of Accuracy', fontsize=10, fontweight='bold')
    ax4.set_title('Model Consistency Across All Systems', fontsize=11, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 2])
    
    gaps = []
    for model in model_names:
        weak_avg = np.mean([weak[sys][model][0] for sys in weak.keys()]) if weak else 0
        strong_avg = np.mean([strong[sys][model][0] for sys in strong.keys()]) if strong else 0
        gaps.append(weak_avg - strong_avg)
    
    colors_gap = ['green' if g > 0 else 'red' for g in gaps]
    ax5.bar(model_names, gaps, alpha=0.7, color=colors_gap, edgecolor='black')
    ax5.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Accuracy Gap', fontsize=10, fontweight='bold')
    ax5.set_title('Weak vs Strong Systems Gap', fontsize=11, fontweight='bold')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.tick_params(axis='x', rotation=45)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax5.grid(axis='y', alpha=0.3)
    
    return fig


def main():
    print("="*70)
    print("DIFFERENTIAL ANALYSIS: ML Models vs Cryptographic Systems")
    print("="*70)
    
    all_results = {}
    for cipher in systems_to_test:
        results = run_bench(cipher_name=cipher, n_samples=2000, msg_len=16)
        all_results[cipher] = {name: (acc, train_t, inf_t, size) 
                              for name, acc, train_t, inf_t, size in results}
    
    print("\n" + "="*70)
    print("Generating differential plots...")
    print("="*70)
    
    fig1 = plot_accuracy_by_category(all_results)
    plt.savefig('diff_accuracy_by_category.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: diff_accuracy_by_category.png")
    
    fig2 = plot_accuracy_diff_matrix(all_results)
    plt.savefig('diff_model_comparison_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: diff_model_comparison_matrix.png")
    
    fig3 = plot_category_performance_heatmap(all_results)
    plt.savefig('diff_category_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: diff_category_heatmap.png")
    
    fig4 = plot_model_effectiveness_across_categories(all_results)
    plt.savefig('diff_model_effectiveness.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: diff_model_effectiveness.png")
    
    fig5 = plot_differential_analysis(all_results)
    plt.savefig('diff_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: diff_comprehensive_analysis.png")
    
    fig6 = plot_timing_analysis(all_results)
    plt.savefig('diff_timing_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: diff_timing_analysis.png")
    
    print("\n" + "="*70)
    print("All differential plots generated successfully!")
    print("="*70)
    
    # Print summary statistics
    weak, semi, strong = categorize_results(all_results)
    print(f"\nSystem Categories:")
    print(f"  Weak Systems:      {len(weak)} systems")
    print(f"  Semi-Weak Systems: {len(semi)} systems")
    print(f"  Strong Systems:    {len(strong)} systems")
    


if __name__ == '__main__':
    main()