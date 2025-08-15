#!/usr/bin/env python3
"""
Project: Bankruptcy Prevention Analysis
Converted from Jupyter Notebook to Python script
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up plotting parameters
plt.style.use('default')
sns.set_palette("husl")

def main():
    """
    Main function to run the bankruptcy prevention analysis
    """
    print("=== Bankruptcy Prevention Analysis ===\n")
    
    # Load and explore the data
    print("Loading data...")
    # Note: Update the file path as needed
    # data = pd.read_csv('your_bankruptcy_data.csv')
    
    # For demonstration, we'll create sample data structure
    # Replace this section with actual data loading
    print("Creating sample data structure for demonstration...")
    
    # Sample data creation (replace with actual data loading)
    np.random.seed(42)
    n_samples = 1000
    
    # Sample features that might indicate bankruptcy risk
    sample_data = {
        'current_ratio': np.random.normal(2.5, 1.0, n_samples),
        'debt_to_equity': np.random.normal(0.6, 0.3, n_samples),
        'return_on_assets': np.random.normal(0.05, 0.15, n_samples),
        'working_capital': np.random.normal(50000, 20000, n_samples),
        'cash_flow': np.random.normal(10000, 15000, n_samples),
        'revenue_growth': np.random.normal(0.03, 0.2, n_samples),
        'profit_margin': np.random.normal(0.1, 0.1, n_samples),
        'quick_ratio': np.random.normal(1.2, 0.5, n_samples)
    }
    
    data = pd.DataFrame(sample_data)
    
    # Create target variable (bankruptcy risk)
    # Higher risk if multiple negative indicators
    risk_score = (
        (data['current_ratio'] < 1.0).astype(int) +
        (data['debt_to_equity'] > 1.0).astype(int) +
        (data['return_on_assets'] < 0).astype(int) +
        (data['cash_flow'] < 0).astype(int) +
        (data['profit_margin'] < 0).astype(int)
    )
    
    data['bankruptcy_risk'] = (risk_score >= 2).astype(int)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Bankruptcy risk distribution:\n{data['bankruptcy_risk'].value_counts()}\n")
    
    # Data exploration
    explore_data(data)
    
    # Feature engineering and preprocessing
    X, y = prepare_features(data)
    
    # Train models
    models = train_models(X, y)
    
    # Generate insights and recommendations
    generate_insights(data, models)
    
    print("\n=== Analysis Complete ===")

def explore_data(data):
    """
    Explore and visualize the dataset
    """
    print("=== Data Exploration ===")
    
    # Basic statistics
    print("Dataset Info:")
    print(data.info())
    print(f"\nDataset Description:\n{data.describe()}")
    
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing Values:\n{missing_values[missing_values > 0]}")
    else:
        print("\nNo missing values found.")
    
    # Correlation analysis
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('/workspace/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Distribution of key features
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    numeric_columns = [col for col in numeric_columns if col != 'bankruptcy_risk']
    
    for i, col in enumerate(numeric_columns[:8]):
        if i < len(axes):
            axes[i].hist(data[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('/workspace/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Risk analysis by feature
    analyze_risk_factors(data)

def analyze_risk_factors(data):
    """
    Analyze risk factors for bankruptcy
    """
    print("\n=== Risk Factor Analysis ===")
    
    # Compare features between high-risk and low-risk companies
    risk_comparison = data.groupby('bankruptcy_risk').agg({
        'current_ratio': ['mean', 'std'],
        'debt_to_equity': ['mean', 'std'],
        'return_on_assets': ['mean', 'std'],
        'cash_flow': ['mean', 'std'],
        'profit_margin': ['mean', 'std']
    }).round(3)
    
    print("Risk Factor Comparison (High Risk vs Low Risk):")
    print(risk_comparison)
    
    # Visualize risk factors
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    key_features = ['current_ratio', 'debt_to_equity', 'return_on_assets', 
                   'cash_flow', 'profit_margin', 'quick_ratio']
    
    for i, feature in enumerate(key_features):
        if i < len(axes):
            data.boxplot(column=feature, by='bankruptcy_risk', ax=axes[i])
            axes[i].set_title(f'{feature} by Bankruptcy Risk')
            axes[i].set_xlabel('Bankruptcy Risk (0=Low, 1=High)')
    
    plt.tight_layout()
    plt.savefig('/workspace/risk_factor_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def prepare_features(data):
    """
    Prepare features for modeling
    """
    print("\n=== Feature Preparation ===")
    
    # Select features (exclude target variable)
    feature_columns = [col for col in data.columns if col != 'bankruptcy_risk']
    X = data[feature_columns]
    y = data['bankruptcy_risk']
    
    # Handle any remaining missing values
    X = X.fillna(X.median())
    
    print(f"Features selected: {list(X.columns)}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def train_models(X, y):
    """
    Train multiple models for bankruptcy prediction
    """
    print("\n=== Model Training ===")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Train and evaluate models
    model_results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for Logistic Regression, original for Random Forest
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{name} Accuracy: {accuracy:.3f}")
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))
        
        # Store results
        model_results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred,
            'test_labels': y_test
        }
        
        # Feature importance (for Random Forest)
        if name == 'Random Forest':
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop Feature Importances for {name}:")
            print(feature_importance.head())
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance['feature'][::-1], feature_importance['importance'][::-1])
            plt.title(f'Feature Importance - {name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('/workspace/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    return model_results

def generate_insights(data, models):
    """
    Generate business insights and recommendations
    """
    print("\n=== Business Insights and Recommendations ===")
    
    # Risk distribution analysis
    risk_stats = data['bankruptcy_risk'].value_counts(normalize=True)
    print(f"Overall bankruptcy risk distribution:")
    print(f"Low Risk: {risk_stats[0]:.1%}")
    print(f"High Risk: {risk_stats[1]:.1%}")
    
    # Key risk indicators
    high_risk_data = data[data['bankruptcy_risk'] == 1]
    low_risk_data = data[data['bankruptcy_risk'] == 0]
    
    print(f"\nKey Risk Indicators:")
    print(f"High-risk companies typically have:")
    print(f"- Current Ratio: {high_risk_data['current_ratio'].mean():.2f} (vs {low_risk_data['current_ratio'].mean():.2f} for low-risk)")
    print(f"- Debt-to-Equity: {high_risk_data['debt_to_equity'].mean():.2f} (vs {low_risk_data['debt_to_equity'].mean():.2f} for low-risk)")
    print(f"- Return on Assets: {high_risk_data['return_on_assets'].mean():.3f} (vs {low_risk_data['return_on_assets'].mean():.3f} for low-risk)")
    
    # Recommendations
    print(f"\n=== Bankruptcy Prevention Recommendations ===")
    print("1. Monitor Current Ratio: Maintain above 1.5 for healthy liquidity")
    print("2. Control Debt Levels: Keep debt-to-equity ratio below 0.8")
    print("3. Focus on Profitability: Ensure positive return on assets")
    print("4. Cash Flow Management: Maintain positive operating cash flow")
    print("5. Regular Financial Health Checks: Implement quarterly risk assessments")
    
    # Early warning system
    print(f"\n=== Early Warning System Thresholds ===")
    print("RED FLAGS (High Risk Indicators):")
    print("- Current Ratio < 1.0")
    print("- Debt-to-Equity > 1.0")
    print("- Negative Return on Assets")
    print("- Negative Cash Flow for 2+ consecutive quarters")
    print("- Declining profit margins")
    
    # Model performance summary
    print(f"\n=== Model Performance Summary ===")
    for name, results in models.items():
        print(f"{name}: {results['accuracy']:.1%} accuracy")

def create_risk_dashboard():
    """
    Create a simple risk assessment dashboard
    """
    print("\n=== Risk Assessment Dashboard ===")
    print("This function would create an interactive dashboard for:")
    print("- Real-time risk monitoring")
    print("- Company financial health scores")
    print("- Alert systems for deteriorating conditions")
    print("- Trend analysis and forecasting")

if __name__ == "__main__":
    main()
    create_risk_dashboard()
    
    print("\n" + "="*50)
    print("IMPORTANT NOTES:")
    print("1. Replace sample data with actual bankruptcy dataset")
    print("2. Adjust feature engineering based on your specific data")
    print("3. Fine-tune model parameters for better performance")
    print("4. Implement proper data validation and error handling")
    print("5. Consider additional models (XGBoost, SVM, etc.)")
    print("="*50)