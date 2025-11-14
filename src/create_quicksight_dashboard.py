#!/usr/bin/env python3
"""
QuickSight Dashboard Creation Script for Fraud Detection Workshop
Creates comprehensive fraud analytics dashboard using participant's generated data
"""

import boto3
import json
import time
from datetime import datetime

class QuickSightDashboardCreator:
    def __init__(self, account_id, region='us-west-2'):
        self.account_id = account_id
        self.region = region
        self.quicksight = boto3.client('quicksight', region_name=region)
        self.data_source_id = f"fraud-data-source-{int(time.time())}"
        self.dataset_id = f"fraud-dataset-{int(time.time())}"
        self.dashboard_id = f"fraud-dashboard-{int(time.time())}"
        
    def create_data_source(self, s3_bucket):
        """Create QuickSight data source pointing to user's S3 data"""
        
        manifest_uri = f"s3://{s3_bucket}/config/quicksight_manifest.json"
        
        data_source_params = {
            'AwsAccountId': self.account_id,
            'DataSourceId': self.data_source_id,
            'Name': 'Fraud Detection Data Source',
            'Type': 'S3',
            'DataSourceParameters': {
                'S3Parameters': {
                    'ManifestFileLocation': {
                        'Bucket': s3_bucket,
                        'Key': 'config/quicksight_manifest.json'
                    }
                }
            },
            # Skip permissions - will be set automatically by QuickSight
        }
        
        try:
            response = self.quicksight.create_data_source(**data_source_params)
            print(f"✅ Data source created: {self.data_source_id}")
            return response
        except Exception as e:
            print(f"❌ Error creating data source: {str(e)}")
            return None
    
    def create_dataset(self):
        """Create QuickSight dataset from the data source"""
        
        dataset_params = {
            'AwsAccountId': self.account_id,
            'DataSetId': self.dataset_id,
            'Name': 'Fraud Detection Dataset',
            'PhysicalTableMap': {
                'fraud-data': {
                    'S3Source': {
                        'DataSourceArn': f"arn:aws:quicksight:{self.region}:{self.account_id}:datasource/{self.data_source_id}",
                        'InputColumns': [
                            {'Name': 'transaction_id', 'Type': 'STRING'},
                            {'Name': 'sender_account', 'Type': 'STRING'},
                            {'Name': 'receiver_account', 'Type': 'STRING'},
                            {'Name': 'amount', 'Type': 'DECIMAL'},
                            {'Name': 'payment_type', 'Type': 'STRING'},
                            {'Name': 'timestamp', 'Type': 'DATETIME'},
                            {'Name': 'hour', 'Type': 'INTEGER'},
                            {'Name': 'is_fraud', 'Type': 'BIT'},
                            {'Name': 'fraud_type', 'Type': 'STRING'},
                            {'Name': 'risk_score', 'Type': 'DECIMAL'},
                            {'Name': 'fraud_indicator', 'Type': 'STRING'},
                            {'Name': 'risk_category', 'Type': 'STRING'},
                            {'Name': 'amount_bucket', 'Type': 'STRING'},
                            {'Name': 'time_category', 'Type': 'STRING'},
                            {'Name': 'day_of_week', 'Type': 'STRING'},
                            {'Name': 'cross_institution', 'Type': 'BIT'},
                            {'Name': 'sender_institution', 'Type': 'STRING'},
                            {'Name': 'receiver_institution', 'Type': 'STRING'},
                            {'Name': 'is_shell', 'Type': 'BIT'},
                            {'Name': 'is_synthetic', 'Type': 'BIT'}
                        ]
                    }
                }
            },
            'ImportMode': 'SPICE',
            # Skip permissions - will be set automatically by QuickSight
        }
        
        try:
            response = self.quicksight.create_data_set(**dataset_params)
            print(f"✅ Dataset created: {self.dataset_id}")
            return response
        except Exception as e:
            print(f"❌ Error creating dataset: {str(e)}")
            return None
    
    def load_dashboard_config(self, config_file='quicksight_fraud_dashboard.json'):
        """Load dashboard configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"❌ Error loading dashboard config: {str(e)}")
            return None
    
    def create_dashboard_from_config(self, config):
        """Create comprehensive fraud detection dashboard from configuration"""
        
        # Build calculated fields
        calculated_fields = []
        if 'calculatedFields' in config:
            for field in config['calculatedFields']:
                calculated_fields.append({
                    'DataSetIdentifier': 'fraud-data',
                    'Name': field['name'],
                    'Expression': field['expression']
                })
        
        # Build basic dashboard definition
        dashboard_definition = {
            'DataSetIdentifierDeclarations': [
                {
                    'DataSetArn': f"arn:aws:quicksight:{self.region}:{self.account_id}:dataset/{self.dataset_id}",
                    'Identifier': 'fraud-data'
                }
            ],
            'Sheets': [],
            'CalculatedFields': calculated_fields
        }
        
        # Add sheets from configuration
        for sheet_config in config.get('sheets', []):
            sheet = {
                'SheetId': sheet_config['sheetId'],
                'Name': sheet_config['name'],
                'Visuals': []
            }
            
            # Add visuals for this sheet
            for visual_config in sheet_config.get('visuals', []):
                visual = self.create_visual_from_config(visual_config)
                if visual:
                    sheet['Visuals'].append(visual)
            
            dashboard_definition['Sheets'].append(sheet)
        
        # Create the dashboard
        dashboard_params = {
            'AwsAccountId': self.account_id,
            'DashboardId': self.dashboard_id,
            'Name': config.get('dashboardName', 'Fraud Detection Dashboard'),
            'Definition': dashboard_definition
            # Skip permissions - will be set automatically by QuickSight
        }
        
        try:
            response = self.quicksight.create_dashboard(**dashboard_params)
            print(f"✅ Dashboard created: {self.dashboard_id}")
            dashboard_url = f"https://{self.region}.quicksight.aws.amazon.com/sn/dashboards/{self.dashboard_id}"
            print(f"🔗 Dashboard URL: {dashboard_url}")
            return response
        except Exception as e:
            print(f"❌ Error creating dashboard: {str(e)}")
            return None
    
    def create_visual_from_config(self, visual_config):
        """Create a visual configuration from JSON specification"""
        
        visual_type = visual_config.get('type', 'KPI')
        
        # Basic visual structure
        visual = {
            'VisualId': visual_config['visualId'],
            'Title': {
                'Visibility': 'VISIBLE',
                'Label': visual_config.get('title', 'Untitled')
            }
        }
        
        # Add visual-specific configuration based on type
        if visual_type == 'PIE_CHART':
            visual['PieChartVisual'] = {
                'VisualId': visual_config['visualId'],
                'Title': visual['Title'],
                'ChartConfiguration': {
                    'FieldWells': {
                        'PieChartAggregatedFieldWells': {
                            'Category': [self.create_field_well(field) for field in visual_config['fieldWells'].get('category', [])],
                            'Values': [self.create_field_well(field) for field in visual_config['fieldWells'].get('values', [])]
                        }
                    }
                }
            }
        elif visual_type == 'KPI':
            visual['KPIVisual'] = {
                'VisualId': visual_config['visualId'],
                'Title': visual['Title'],
                'ChartConfiguration': {
                    'FieldWells': {
                        'Values': [self.create_field_well(field) for field in visual_config['fieldWells'].get('values', [])]
                    }
                }
            }
        elif visual_type == 'SANKEY_DIAGRAM':
            visual['SankeyDiagramVisual'] = {
                'VisualId': visual_config['visualId'],
                'Title': visual['Title'],
                'ChartConfiguration': {
                    'FieldWells': {
                        'SankeyDiagramAggregatedFieldWells': {
                            'Source': [self.create_field_well(field) for field in visual_config['fieldWells'].get('source', [])],
                            'Destination': [self.create_field_well(field) for field in visual_config['fieldWells'].get('destination', [])],
                            'Weight': [self.create_field_well(field) for field in visual_config['fieldWells'].get('weight', [])]
                        }
                    }
                }
            }
        
        return visual
    
    def create_field_well(self, field_name):
        """Create a field well configuration"""
        return {
            'CategoricalDimensionField': {
                'FieldId': field_name.lower().replace('_', '-'),
                'Column': {
                    'DataSetIdentifier': 'fraud-data',
                    'ColumnName': field_name
                }
            }
        }
    
    def create_simple_dashboard(self):
        """Create a simplified dashboard with basic visualizations"""
        
        dashboard_definition = {
            'DataSetIdentifierDeclarations': [
                {
                    'DataSetArn': f"arn:aws:quicksight:{self.region}:{self.account_id}:dataset/{self.dataset_id}",
                    'Identifier': 'fraud-data'
                }
            ],
            'Sheets': [
                {
                    'SheetId': 'executive-summary',
                    'Name': 'Executive Summary',
                    'Visuals': [
                        {
                            'KPIVisual': {
                                'VisualId': 'fraud-transactions-kpi',
                                'Title': {'Visibility': 'VISIBLE', 'Label': 'Total Fraud Transactions'},
                                'ChartConfiguration': {
                                    'FieldWells': {
                                        'Values': [
                                            {
                                                'NumericalMeasureField': {
                                                    'FieldId': 'fraud-count',
                                                    'Column': {'DataSetIdentifier': 'fraud-data', 'ColumnName': 'transaction_id'},
                                                    'AggregationFunction': {'SimpleNumericalAggregation': 'COUNT'}
                                                }
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    ]
                }
            ],
            'CalculatedFields': [
                {
                    'DataSetIdentifier': 'fraud-data',
                    'Name': 'FraudRate',
                    'Expression': '(countIf({fraud_indicator} = "Fraud") / count({transaction_id})) * 100'
                }
            ]
        }
        
        dashboard_params = {
            'AwsAccountId': self.account_id,
            'DashboardId': self.dashboard_id,
            'Name': 'Fraud Detection Analytics - Workshop',
            'Definition': dashboard_definition
            # Skip permissions - will be set automatically by QuickSight
        }
        
        try:
            response = self.quicksight.create_dashboard(**dashboard_params)
            print(f"✅ Dashboard created: {self.dashboard_id}")
            dashboard_url = f"https://{self.region}.quicksight.aws.amazon.com/sn/dashboards/{self.dashboard_id}"
            print(f"🔗 Dashboard URL: {dashboard_url}")
            return response
        except Exception as e:
            print(f"❌ Error creating dashboard: {str(e)}")
            return None
    
    def create_complete_dashboard(self, s3_bucket):
        """Create complete fraud detection dashboard workflow"""
        
        print("🚀 Creating QuickSight Fraud Detection Dashboard...")
        print(f"📊 Using data from: s3://{s3_bucket}")
        
        # Step 1: Create data source
        if not self.create_data_source(s3_bucket):
            return False
        
        # Wait for data source to be ready
        time.sleep(5)
        
        # Step 2: Create dataset
        if not self.create_dataset():
            return False
        
        # Wait for dataset to be ready
        time.sleep(10)
        
        # Step 3: Try to create dashboard from config, fallback to simple
        config = self.load_dashboard_config()
        if config:
            print("📋 Using comprehensive dashboard configuration...")
            dashboard_result = self.create_dashboard_from_config(config)
        else:
            print("📋 Using simplified dashboard configuration...")
            dashboard_result = self.create_simple_dashboard()
        
        if not dashboard_result:
            return False
        
        print("✅ Fraud Detection Dashboard created successfully!")
        print(f"🎯 Dashboard ID: {self.dashboard_id}")
        
        return True

def main():
    """Main function for workshop execution"""
    
    # Get AWS account info
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    
    # QuickSight bucket name (matches notebook pattern)
    s3_bucket = f"{account_id}-quicksight-fraud-data"
    
    print(f"🎯 Creating dashboard for account: {account_id}")
    print(f"📁 Using S3 bucket: {s3_bucket}")
    
    # Create dashboard
    creator = QuickSightDashboardCreator(account_id)
    success = creator.create_complete_dashboard(s3_bucket)
    
    if success:
        print("\n🎉 Workshop Dashboard Ready!")
        print("👉 Go to QuickSight Console to explore your fraud analytics")
    else:
        print("\n❌ Dashboard creation failed")
        print("👉 Check AWS permissions and try manual creation")

if __name__ == "__main__":
    main()