#!/usr/bin/env python3
import pandas as pd
import json
from datetime import datetime

def prepare_quicksight_data():
    """Prepare and enhance data for QuickSight visualization"""
    
    # Load datasets
    print("Loading datasets...")
    transactions = pd.read_csv('transactions.csv')
    accounts = pd.read_csv('accounts.csv')
    institutions = pd.read_csv('institutions.csv')
    
    # Create enhanced transaction dataset
    print("Creating enhanced transaction dataset...")
    
    # Join transactions with accounts and institutions
    enhanced_tx = transactions.merge(accounts, left_on='sender_account', right_on='account_id', suffixes=('', '_sender'))
    enhanced_tx = enhanced_tx.merge(accounts, left_on='receiver_account', right_on='account_id', suffixes=('', '_receiver'))
    enhanced_tx = enhanced_tx.merge(institutions, left_on='sender_institution', right_on='institution_id', suffixes=('', '_sender_inst'))
    enhanced_tx = enhanced_tx.merge(institutions, left_on='receiver_institution', right_on='institution_id', suffixes=('', '_receiver_inst'))
    
    # Add calculated fields for QuickSight
    enhanced_tx['fraud_indicator'] = enhanced_tx['is_fraud'].map({True: 'Fraud', False: 'Normal'})
    enhanced_tx['risk_category'] = pd.cut(enhanced_tx['risk_score'], 
                                         bins=[0, 25, 50, 75, 100], 
                                         labels=['Low', 'Medium', 'High', 'Critical'])
    
    enhanced_tx['amount_bucket'] = pd.cut(enhanced_tx['amount'],
                                         bins=[0, 1000, 10000, 50000, 100000, float('inf')],
                                         labels=['<$1K', '$1K-$10K', '$10K-$50K', '$50K-$100K', '$100K+'])
    
    enhanced_tx['time_category'] = enhanced_tx['hour'].apply(lambda x: 
        'Business Hours' if 9 <= x <= 17 else
        'Extended Hours' if 6 <= x <= 21 else
        'Off Hours')
    
    # Parse timestamp for better date handling
    enhanced_tx['timestamp'] = pd.to_datetime(enhanced_tx['timestamp'])
    enhanced_tx['date'] = enhanced_tx['timestamp'].dt.date
    enhanced_tx['day_of_week'] = enhanced_tx['timestamp'].dt.day_name()
    enhanced_tx['month'] = enhanced_tx['timestamp'].dt.month_name()
    
    # Cross-institution flag
    enhanced_tx['cross_institution'] = enhanced_tx['sender_institution'] != enhanced_tx['receiver_institution']
    
    # Select relevant columns for QuickSight
    quicksight_columns = [
        'transaction_id', 'sender_account', 'receiver_account',
        'sender_institution', 'receiver_institution', 'amount', 'payment_type',
        'timestamp', 'date', 'hour', 'day_of_week', 'month',
        'is_fraud', 'fraud_type', 'risk_score', 'fraud_indicator',
        'risk_category', 'amount_bucket', 'time_category', 'cross_institution',
        'balance', 'balance_receiver', 'is_shell', 'is_synthetic',
        'is_shell_receiver', 'is_synthetic_receiver',
        'name', 'type', 'name_receiver_inst', 'type_receiver_inst'
    ]
    
    # Create final dataset
    quicksight_data = enhanced_tx[quicksight_columns].copy()
    
    # Save enhanced dataset
    quicksight_data.to_csv('quicksight_fraud_data.csv', index=False)
    print(f"Enhanced dataset saved: {len(quicksight_data)} transactions")
    
    # Create fraud summary for KPIs
    create_fraud_summary(quicksight_data)
    
    # Create S3 manifest file
    create_s3_manifest()
    
    print("QuickSight data preparation complete!")

def create_fraud_summary(data):
    """Create fraud summary statistics"""
    
    summary = {
        'total_transactions': len(data),
        'fraud_transactions': len(data[data['is_fraud'] == True]),
        'fraud_rate': (len(data[data['is_fraud'] == True]) / len(data)) * 100,
        'total_fraud_amount': data[data['is_fraud'] == True]['amount'].sum(),
        'avg_fraud_amount': data[data['is_fraud'] == True]['amount'].mean(),
        'max_fraud_amount': data[data['is_fraud'] == True]['amount'].max(),
        'avg_risk_score': data['risk_score'].mean(),
        'fraud_by_type': data[data['is_fraud'] == True]['fraud_type'].value_counts().to_dict(),
        'fraud_by_payment_type': data[data['is_fraud'] == True]['payment_type'].value_counts().to_dict(),
        'fraud_by_hour': data[data['is_fraud'] == True]['hour'].value_counts().sort_index().to_dict()
    }
    
    # Save summary as JSON
    with open('fraud_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("Fraud summary statistics:")
    print(f"- Total transactions: {summary['total_transactions']:,}")
    print(f"- Fraud transactions: {summary['fraud_transactions']:,}")
    print(f"- Fraud rate: {summary['fraud_rate']:.2f}%")
    print(f"- Total fraud amount: ${summary['total_fraud_amount']:,.2f}")

def create_s3_manifest(bucket_name=None):
    """Create S3 manifest file for QuickSight"""
    
    # Use provided bucket name or default placeholder
    bucket_uri = f"s3://{bucket_name}/enhanced-data/" if bucket_name else "s3://ACCOUNT-quicksight-fraud-data/enhanced-data/"
    
    manifest = {
        "fileLocations": [
            {
                "URIPrefixes": [
                    bucket_uri
                ]
            }
        ],
        "globalUploadSettings": {
            "format": "CSV",
            "delimiter": ",",
            "textqualifier": "\"",
            "containsHeader": "true"
        }
    }
    
    with open('quicksight_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("S3 manifest file created: quicksight_manifest.json")

def create_dashboard_config():
    """Create QuickSight dashboard configuration"""
    
    config = {
        "dashboardId": "fraud-detection-dashboard",
        "name": "Fraud Detection Analytics - 100K Transactions",
        "description": "Comprehensive fraud analysis across all 9 fraud types",
        "sheets": [
            {
                "sheetId": "executive-summary",
                "name": "Executive Summary",
                "description": "High-level fraud metrics and trends"
            },
            {
                "sheetId": "fraud-types",
                "name": "Fraud Type Analysis", 
                "description": "Detailed analysis by fraud pattern"
            },
            {
                "sheetId": "network-analysis",
                "name": "Network Analysis",
                "description": "Institution and account relationships"
            },
            {
                "sheetId": "operational",
                "name": "Operational Monitoring",
                "description": "Real-time fraud monitoring"
            }
        ],
        "calculatedFields": [
            {
                "name": "FraudRate",
                "expression": "sum(ifelse({is_fraud} = 'True', 1, 0)) / count({transaction_id}) * 100"
            },
            {
                "name": "RiskLevel", 
                "expression": "ifelse({risk_score} >= 80, 'Critical', ifelse({risk_score} >= 60, 'High', ifelse({risk_score} >= 40, 'Medium', 'Low')))"
            }
        ],
        "parameters": [
            {
                "name": "RiskThreshold",
                "type": "INTEGER",
                "defaultValue": 70
            },
            {
                "name": "DateRange",
                "type": "DATETIME",
                "defaultValue": "LAST_30_DAYS"
            }
        ]
    }
    
    with open('dashboard_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Dashboard configuration created: dashboard_config.json")

if __name__ == "__main__":
    prepare_quicksight_data()
    create_dashboard_config()