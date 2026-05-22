import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def generate_dashboard():
    # 1. Load Data
    csv_path = "validation_results.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] Could not find '{csv_path}'. Run 'batch_validator.py' first!")
        return

    df = pd.read_csv(csv_path)
    
    # 2. Setup Dashboard
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('Drone Swarm Defense System - Final Validation Report', fontsize=16, weight='bold')
    
    # --- PLOT 1: Win Rate Comparison (Pie Charts) ---
    ax1 = plt.subplot(2, 2, 1)
    
    # Calculate Wins
    scen_a_wins = len(df[(df['Scenario'] == 'A') & (df['Outcome'] == 'WIN')])
    scen_a_total = len(df[df['Scenario'] == 'A'])
    
    scen_b_wins = len(df[(df['Scenario'] == 'B') & (df['Outcome'] == 'WIN')])
    scen_b_total = len(df[df['Scenario'] == 'B'])
    
    # Data for Plot
    labels = ['Scenario A (Parity)', 'Scenario B (Asymmetric)']
    # Handle case where total is 0 to avoid division by zero
    if scen_a_total == 0: scen_a_total = 1
    if scen_b_total == 0: scen_b_total = 1
    
    win_rates = [(scen_a_wins/scen_a_total)*100, (scen_b_wins/scen_b_total)*100]
    colors = ['#66b3ff', '#ff9999']
    
    ax1.bar(labels, win_rates, color=colors, alpha=0.8)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('Mission Success Rate')
    for i, v in enumerate(win_rates):
        ax1.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

    # --- PLOT 2: Friendly Losses (Box Plot) ---
    ax2 = plt.subplot(2, 2, 2)
    data_a = df[df['Scenario']=='A']['Friendly_Losses']
    data_b = df[df['Scenario']=='B']['Friendly_Losses']
    
    ax2.boxplot([data_a, data_b], labels=['Scenario A', 'Scenario B'], patch_artist=True)
    ax2.set_ylabel('Drones Lost (out of 10)')
    ax2.set_title('Casualty Analysis (The Cost of War)')
    ax2.grid(True, axis='y', alpha=0.3)

    # --- PLOT 3: Unattended Time (Violations) ---
    ax3 = plt.subplot(2, 1, 2)
    
    # Scatter plot: Violations vs Friendly Losses
    # Does losing drones make us safer? (Optimization Curve)
    scat = ax3.scatter(df['Friendly_Losses'], df['Violations'], c=df['Steps'], cmap='viridis', alpha=0.7, s=50)
    
    cbar = plt.colorbar(scat, ax=ax3)
    cbar.set_label('Duration (Frames)')
    
    ax3.set_xlabel('Friendly Drones Lost')
    ax3.set_ylabel('Unattended Violations (Risk)')
    ax3.set_title('Efficiency Analysis: Does Sacrifice Buy Safety?')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save to file for verification
    plt.savefig('dashboard.png')
    print("[INFO] Dashboard generated and saved to dashboard.png.")
    # plt.show() # Commented out for headless environment

if __name__ == "__main__":
    generate_dashboard()
