import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

def create_learning_visualizations(rewards, promo_usage, save_path="results/"):
    """Create comprehensive learning visualizations"""
    
    import os
    os.makedirs(save_path, exist_ok=True)
    
    episodes = range(1, len(rewards) + 1)
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    # 1. Learning Curve with Moving Average
    plt.figure(figsize=(14, 10))
    
    # Main learning curve
    plt.subplot(2, 2, 1)
    plt.plot(episodes, rewards, alpha=0.3, color='lightblue', label='Individual Episodes')
    
    # Moving averages
    window_50 = uniform_filter1d(rewards, size=50, mode='nearest')
    window_100 = uniform_filter1d(rewards, size=100, mode='nearest')
    
    plt.plot(episodes, window_50, color='blue', linewidth=2, label='50-Episode Moving Average')
    plt.plot(episodes, window_100, color='red', linewidth=2, label='100-Episode Moving Average')
    
    plt.title('🎯 Agent Learning Curve: Reward Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotations for key insights
    plt.annotate(f'Peak Performance\n{max(rewards):.1f} reward', 
                xy=(np.argmax(rewards), max(rewards)), 
                xytext=(len(rewards)*0.7, max(rewards)*0.8),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.annotate(f'Final Average\n{np.mean(rewards[-50:]):.1f} reward', 
                xy=(len(rewards)-25, np.mean(rewards[-50:])), 
                xytext=(len(rewards)*0.7, max(rewards)*0.6),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # 2. Promotion Usage Over Time
    plt.subplot(2, 2, 2)
    
    # Promotion usage with moving average
    promo_ma = uniform_filter1d(promo_usage, size=50, mode='nearest')
    
    plt.bar(episodes, promo_usage, alpha=0.6, color='orange', label='Promotions per Episode')
    plt.plot(episodes, promo_ma, color='red', linewidth=3, label='50-Episode Moving Average')
    
    plt.title('🎁 Promotion Usage Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Promotions Used')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Performance Phases Analysis
    plt.subplot(2, 2, 3)
    
    # Split into phases
    phase_size = len(rewards) // 5
    phases = []
    phase_labels = []
    
    for i in range(5):
        start_idx = i * phase_size
        end_idx = (i + 1) * phase_size if i < 4 else len(rewards)
        phase_rewards = rewards[start_idx:end_idx]
        phases.append(np.mean(phase_rewards))
        phase_labels.append(f'Phase {i+1}\n({start_idx+1}-{end_idx})')
    
    bars = plt.bar(range(5), phases, color=['red', 'orange', 'yellow', 'lightgreen', 'green'], 
                   alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, phases)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('📊 Performance by Learning Phase', fontsize=14, fontweight='bold')
    plt.xlabel('Learning Phase')
    plt.ylabel('Average Reward')
    plt.xticks(range(5), [f'Phase {i+1}' for i in range(5)])
    plt.grid(True, alpha=0.3)
    
    # 4. Success Rate Analysis
    plt.subplot(2, 2, 4)
    
    # Calculate success rates (positive rewards) over time
    window_size = 50
    success_rates = []
    episode_windows = []
    
    for i in range(window_size, len(rewards) + 1):
        window_rewards = rewards[i-window_size:i]
        success_rate = np.mean([r > 0 for r in window_rewards]) * 100
        success_rates.append(success_rate)
        episode_windows.append(i)
    
    plt.plot(episode_windows, success_rates, color='green', linewidth=3, label='Success Rate')
    plt.fill_between(episode_windows, success_rates, alpha=0.3, color='green')
    
    plt.title('✅ Success Rate Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add final success rate annotation
    final_success = success_rates[-1]
    plt.annotate(f'Final: {final_success:.1f}%', 
                xy=(episode_windows[-1], final_success), 
                xytext=(episode_windows[-1]*0.7, final_success-10),
                arrowprops=dict(arrowstyle='->', color='darkgreen'),
                fontsize=12, ha='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_path}learning_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Create a summary comparison chart
    create_summary_chart(rewards, promo_usage, save_path)
    
    # 3. Create detailed performance metrics
    create_performance_metrics(rewards, promo_usage, save_path)

def create_summary_chart(rewards, promo_usage, save_path):
    """Create a summary comparison chart"""
    
    plt.figure(figsize=(12, 8))
    
    # Before vs After comparison
    early_rewards = rewards[:100]
    late_rewards = rewards[-100:]
    
    metrics = ['Average Reward', 'Success Rate (%)', 'Max Reward', 'Promotion Usage (%)']
    
    early_values = [
        np.mean(early_rewards),
        np.mean([r > 0 for r in early_rewards]) * 100,
        max(early_rewards),
        np.mean([p > 0 for p in promo_usage[:100]]) * 100
    ]
    
    late_values = [
        np.mean(late_rewards),
        np.mean([r > 0 for r in late_rewards]) * 100,
        max(late_rewards),
        np.mean([p > 0 for p in promo_usage[-100:]]) * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, early_values, width, label='Early Episodes (1-100)', 
                    color='lightcoral', alpha=0.8)
    bars2 = plt.bar(x + width/2, late_values, width, label='Late Episodes (401-500)', 
                    color='lightgreen', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('🏆 Early vs Late Performance Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_metrics(rewards, promo_usage, save_path):
    """Create detailed performance metrics visualization"""
    
    plt.figure(figsize=(15, 10))
    
    # 1. Reward distribution histogram
    plt.subplot(2, 3, 1)
    plt.hist(rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(rewards):.2f}')
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Promotion usage distribution
    plt.subplot(2, 3, 2)
    promo_counts = np.bincount(promo_usage)
    plt.bar(range(len(promo_counts)), promo_counts, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Promotion Usage Distribution')
    plt.xlabel('Number of Promotions per Episode')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 3. Learning trend
    plt.subplot(2, 3, 3)
    window_size = 100
    trends = []
    for i in range(window_size, len(rewards)):
        trend = np.polyfit(range(window_size), rewards[i-window_size:i], 1)[0]
        trends.append(trend)
    
    plt.plot(range(window_size, len(rewards)), trends, color='purple', linewidth=2)
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.title('Learning Trend (Slope)')
    plt.xlabel('Episode')
    plt.ylabel('Trend (Reward/Episode)')
    plt.grid(True, alpha=0.3)
    
    # 4. Cumulative rewards
    plt.subplot(2, 3, 4)
    cumulative_rewards = np.cumsum(rewards)
    plt.plot(cumulative_rewards, color='darkgreen', linewidth=2)
    plt.title('Cumulative Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward Accumulated')
    plt.grid(True, alpha=0.3)
    
    # 5. Best episodes analysis
    plt.subplot(2, 3, 5)
    top_episodes = np.argsort(rewards)[-20:]  # Top 20 episodes
    plt.scatter(top_episodes, [rewards[i] for i in top_episodes], 
               c='gold', s=100, alpha=0.8, edgecolors='black')
    plt.title('Top 20 Performance Episodes')
    plt.xlabel('Episode Number')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    # 6. Performance consistency
    plt.subplot(2, 3, 6)
    window_size = 50
    consistency = []
    for i in range(window_size, len(rewards)):
        window_std = np.std(rewards[i-window_size:i])
        consistency.append(window_std)
    
    plt.plot(range(window_size, len(rewards)), consistency, color='brown', linewidth=2)
    plt.title('Performance Consistency (Lower = More Consistent)')
    plt.xlabel('Episode')
    plt.ylabel('Standard Deviation')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}detailed_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_all_visualizations(rewards, promo_usage):
    """Generate all visualizations and save them"""
    print("📊 Generating comprehensive learning visualizations...")
    
    # Create results directory
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Generate all plots
    create_learning_visualizations(rewards, promo_usage)
    
    print("✅ All visualizations saved to 'results/' directory:")
    print("  📈 learning_analysis.png - Main learning curves")
    print("  📊 performance_comparison.png - Before/after comparison") 
    print("  📋 detailed_metrics.png - Detailed analysis")
    
    # Print summary statistics for presentation
    print(f"\n🎯 KEY PRESENTATION STATISTICS:")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Best Episode: {max(rewards):.2f}")
    print(f"Success Rate: {np.mean([r > 0 for r in rewards])*100:.1f}%")
    print(f"Promotion Usage: {np.mean([p > 0 for p in promo_usage])*100:.1f}% of episodes")
    print(f"Total Episodes: {len(rewards)}")
    print(f"Improvement: Consistent {np.mean(rewards[-50:]):.1f} reward in final 50 episodes")

if __name__ == "__main__":
    # Example usage - you'll call this with your actual data
    print("Import this module and call generate_all_visualizations(rewards, promo_usage)") 