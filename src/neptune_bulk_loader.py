import pandas as pd
import boto3
import json
import time
from typing import Dict, List, Optional
import os

class NeptuneBulkLoader:
    def __init__(self, neptune_endpoint: str, s3_bucket: str, neptune_role_arn: str):
        self.neptune_endpoint = neptune_endpoint
        self.s3_bucket = s3_bucket
        self.neptune_role_arn = neptune_role_arn
        self.s3_client = boto3.client('s3')
        # Use requests for Neptune loader API since boto3 doesn't have direct support
        import requests
        self.session = boto3.Session()
        self.region = self.session.region_name
        
    def prepare_bulk_load_data(self, data_dir: str = "enhanced_output", output_dir: str = "bulk_load_data"):
        """Convert enhanced fraud data to Neptune bulk load format"""
        print("🔄 Preparing data for Neptune bulk loading...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load generated data
        institutions_df = pd.read_csv(f'{data_dir}/institutions.csv')
        accounts_df = pd.read_csv(f'{data_dir}/accounts.csv')
        transactions_df = pd.read_csv(f'{data_dir}/transactions.csv')
        
        # Prepare vertices
        self._prepare_institution_vertices(institutions_df, output_dir)
        self._prepare_account_vertices(accounts_df, output_dir)
        self._prepare_fraud_pattern_vertices(transactions_df, output_dir)
        
        # Prepare edges
        self._prepare_institution_edges(accounts_df, output_dir)
        self._prepare_transaction_edges(transactions_df, output_dir)
        
        print(f"✅ Bulk load data prepared in {output_dir}/")
        return output_dir
        
    def _prepare_institution_vertices(self, df: pd.DataFrame, output_dir: str):
        """Prepare institution vertices for bulk loading"""
        vertices = []
        for _, row in df.iterrows():
            vertices.append({
                '~id': row['institution_id'],
                '~label': 'Institution',
                'name:String': row['name'],
                'type:String': row['type'],
                'channels:String': row['channels']
            })
        
        vertices_df = pd.DataFrame(vertices)
        vertices_df.to_csv(f'{output_dir}/institutions_vertices.csv', index=False)
        print(f"  Created institutions_vertices.csv ({len(vertices)} vertices)")
        
    def _prepare_account_vertices(self, df: pd.DataFrame, output_dir: str):
        """Prepare account vertices for bulk loading"""
        vertices = []
        for _, row in df.iterrows():
            risk_category = 'normal'
            if row['is_shell']:
                risk_category = 'shell_account'
            elif row['is_synthetic']:
                risk_category = 'synthetic_identity'
                
            vertices.append({
                '~id': row['account_id'],
                '~label': 'Account',
                'account_type:String': row['account_type'],
                'balance:Double': row['balance'],
                'is_shell:Boolean': row['is_shell'],
                'is_synthetic:Boolean': row['is_synthetic'],
                'risk_category:String': risk_category
            })
        
        vertices_df = pd.DataFrame(vertices)
        vertices_df.to_csv(f'{output_dir}/accounts_vertices.csv', index=False)
        print(f"  Created accounts_vertices.csv ({len(vertices)} vertices)")
        
    def _prepare_fraud_pattern_vertices(self, df: pd.DataFrame, output_dir: str):
        """Prepare fraud pattern vertices for bulk loading"""
        fraud_df = df[df['is_fraud'] == True]
        if len(fraud_df) == 0:
            return
            
        # Group by fraud type
        fraud_groups = fraud_df.groupby('fraud_type')
        vertices = []
        
        for fraud_type, group in fraud_groups:
            pattern_id = f"PATTERN_{fraud_type.upper()}"
            vertices.append({
                '~id': pattern_id,
                '~label': 'FraudPattern',
                'pattern_type:String': fraud_type,
                'transaction_count:Int': len(group),
                'total_amount:Double': group['amount'].sum(),
                'avg_amount:Double': group['amount'].mean(),
                'avg_risk_score:Double': group['risk_score'].mean()
            })
        
        if vertices:
            vertices_df = pd.DataFrame(vertices)
            vertices_df.to_csv(f'{output_dir}/fraud_patterns_vertices.csv', index=False)
            print(f"  Created fraud_patterns_vertices.csv ({len(vertices)} vertices)")
            
    def _prepare_institution_edges(self, df: pd.DataFrame, output_dir: str):
        """Prepare institution-account edges for bulk loading"""
        edges = []
        for _, row in df.iterrows():
            edges.append({
                '~id': f"REL_{row['account_id']}_{row['institution_id']}",
                '~from': row['account_id'],
                '~to': row['institution_id'],
                '~label': 'BELONGS_TO',
                'relationship_type:String': 'account_holder'
            })
        
        edges_df = pd.DataFrame(edges)
        edges_df.to_csv(f'{output_dir}/institution_edges.csv', index=False)
        print(f"  Created institution_edges.csv ({len(edges)} edges)")
        
    def _prepare_transaction_edges(self, df: pd.DataFrame, output_dir: str):
        """Prepare transaction edges for bulk loading"""
        edges = []
        for _, row in df.iterrows():
            edge = {
                '~id': row['transaction_id'],
                '~from': row['sender_account'],
                '~to': row['receiver_account'],
                '~label': 'PAYMENT',
                'transaction_id:String': row['transaction_id'],
                'amount:Double': row['amount'],
                'payment_type:String': row['payment_type'],
                'timestamp:String': row['timestamp'],
                'hour:Int': row['hour'],
                'is_fraud:Boolean': row['is_fraud'],
                'risk_score:Double': row['risk_score']
            }
            
            if pd.notna(row['fraud_type']) and row['fraud_type']:
                edge['fraud_type:String'] = row['fraud_type']
            
            # Add AI-specific fields if they exist
            if 'ai_confidence' in row and pd.notna(row['ai_confidence']):
                edge['ai_confidence:Double'] = row['ai_confidence']
            
            if 'pattern_similarity' in row and pd.notna(row['pattern_similarity']):
                edge['pattern_similarity:Double'] = row['pattern_similarity']
            
            if 'generation_method' in row and pd.notna(row['generation_method']):
                edge['generation_method:String'] = row['generation_method']
                
            edges.append(edge)
        
        edges_df = pd.DataFrame(edges)
        edges_df.to_csv(f'{output_dir}/transactions_edges.csv', index=False)
        print(f"  Created transactions_edges.csv ({len(edges)} edges)")
        
    def _ensure_bucket_exists(self):
        """Create S3 bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.s3_bucket)
            print(f"✅ S3 bucket {self.s3_bucket} exists")
        except:
            print(f"🪣 Creating S3 bucket {self.s3_bucket}...")
            try:
                # Get current region
                region = boto3.Session().region_name
                if region == 'us-east-1':
                    # us-east-1 doesn't need LocationConstraint
                    self.s3_client.create_bucket(Bucket=self.s3_bucket)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.s3_bucket,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                print(f"✅ Created S3 bucket {self.s3_bucket}")
            except Exception as e:
                print(f"❌ Failed to create bucket: {str(e)}")
                raise
    
    def upload_to_s3(self, local_dir: str, s3_prefix: str = "fraud-data"):
        """Upload bulk load files to S3"""
        # Ensure bucket exists first
        self._ensure_bucket_exists()
        
        print(f"📤 Uploading files to s3://{self.s3_bucket}/{s3_prefix}/")
        
        uploaded_files = []
        for filename in os.listdir(local_dir):
            if filename.endswith('.csv'):
                local_path = os.path.join(local_dir, filename)
                s3_key = f"{s3_prefix}/{filename}"
                
                print(f"  Uploading {filename}...")
                self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                uploaded_files.append(s3_key)
                
                # Also upload to quicksight-data bucket if it exists
                try:
                    quicksight_bucket = f"{boto3.client('sts').get_caller_identity()['Account']}-quicksight-fraud-data"
                    quicksight_key = f"raw-data/{filename}"
                    self.s3_client.upload_file(local_path, quicksight_bucket, quicksight_key)
                    print(f"  Also uploaded to QuickSight bucket: {quicksight_key}")
                except Exception as e:
                    print(f"  Note: QuickSight bucket not available ({str(e)})")
                
        print(f"✅ Uploaded {len(uploaded_files)} files to S3")
        print(f"💡 Note: Neptune will delete these files after bulk loading")
        print(f"📊 Files also copied to QuickSight bucket for dashboard creation")
        return uploaded_files
        
    def start_bulk_load(self, s3_prefix: str = "fraud-data", parallelism: str = "MEDIUM") -> str:
        """Start Neptune bulk load job using REST API"""
        print("🚀 Starting Neptune bulk load job...")
        
        source_uri = f"s3://{self.s3_bucket}/{s3_prefix}/"
        
        try:
            import requests
            import json
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            
            # Prepare the request
            url = f"https://{self.neptune_endpoint}:8182/loader"
            
            payload = {
                'source': source_uri,
                'format': 'csv',
                'region': self.region,
                'iamRoleArn': self.neptune_role_arn,
                'mode': 'NEW',
                'failOnError': False,
                'parallelism': parallelism
            }
            
            # Convert to JSON
            json_payload = json.dumps(payload)
            
            # Create AWS request and sign it
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            request = AWSRequest(method='POST', url=url, data=json_payload, headers=headers)
            SigV4Auth(self.session.get_credentials(), 'neptune-db', self.region).add_auth(request)
            
            # Make the request
            response = requests.post(url, data=json_payload, headers=dict(request.headers))
            
            print(f"Response status: {response.status_code}")
            if response.status_code != 200:
                print(f"Response text: {response.text}")
            
            response.raise_for_status()
            
            result = response.json()
            load_id = result['payload']['loadId']
            print(f"✅ Bulk load job started with ID: {load_id}")
            return load_id
            
        except Exception as e:
            print(f"❌ Failed to start bulk load: {str(e)}")
            raise
            
    def monitor_bulk_load(self, load_id: str, check_interval: int = 30) -> bool:
        """Monitor bulk load job progress using REST API"""
        print(f"👀 Monitoring bulk load job {load_id}...")
        
        while True:
            try:
                import requests
                from botocore.auth import SigV4Auth
                from botocore.awsrequest import AWSRequest
                
                # Prepare the request
                url = f"https://{self.neptune_endpoint}:8182/loader/{load_id}"
                
                # Create AWS request and sign it
                request = AWSRequest(method='GET', url=url)
                SigV4Auth(self.session.get_credentials(), 'neptune-db', self.region).add_auth(request)
                
                # Make the request
                response = requests.get(url, headers=dict(request.headers))
                response.raise_for_status()
                
                result = response.json()
                status = result['payload']['overallStatus']['status']
                
                print(f"  Status: {status}")
                
                if status == 'LOAD_COMPLETED':
                    print("✅ Bulk load completed successfully!")
                    self._print_load_summary(result['payload'])
                    return True
                elif status in ['LOAD_FAILED', 'LOAD_CANCELLED']:
                    print(f"❌ Bulk load failed with status: {status}")
                    self._print_load_errors(result['payload'])
                    return False
                elif status in ['LOAD_IN_PROGRESS', 'LOAD_DATA_IN_PROGRESS']:
                    time.sleep(check_interval)
                else:
                    print(f"⏳ Waiting... Status: {status}")
                    time.sleep(check_interval)
                    
            except Exception as e:
                print(f"❌ Error monitoring load: {str(e)}")
                return False
                
    def _print_load_summary(self, payload: Dict):
        """Print bulk load summary"""
        stats = payload.get('overallStatus', {})
        print("\n📊 Load Summary:")
        total_records = stats.get('totalRecords', 0)
        duplicates = stats.get('totalDuplicates', 0)
        records_loaded = total_records - duplicates if total_records and duplicates else total_records
        
        print(f"  Total records processed: {total_records:,}")
        print(f"  Records successfully loaded: {records_loaded:,}")
        if duplicates > 0:
            print(f"  Duplicate records skipped: {duplicates:,}")
        print(f"  Time elapsed: {stats.get('timeElapsedSeconds', 'N/A')} seconds")
        
        # Note about Neptune API limitations
        if records_loaded == 0 and total_records > 0:
            print("\n📝 Note: Neptune's load statistics may not reflect actual data loaded.")
            print("   Run queries to verify data was loaded successfully.")
        
    def _print_load_errors(self, payload: Dict):
        """Print bulk load errors"""
        errors = payload.get('overallStatus', {}).get('errors', [])
        if errors:
            print("\n❌ Load Errors:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
                
    def bulk_load_enhanced_fraud_data(self, data_dir: str = "enhanced_output") -> bool:
        """Complete bulk load process for enhanced fraud data"""
        try:
            # Prepare data
            bulk_dir = self.prepare_bulk_load_data(data_dir)
            
            # Upload to S3
            self.upload_to_s3(bulk_dir)
            
            # Start bulk load
            load_id = self.start_bulk_load()
            
            # Monitor progress
            success = self.monitor_bulk_load(load_id)
            
            if success:
                print("\n🎉 Enhanced fraud network successfully loaded into Neptune!")
                print("\n📊 Data also copied to QuickSight bucket for dashboard creation.")
                print("\n🔍 Next steps:")
                print("   • Verify data in Neptune using the analytics notebook")
                print("   • Deploy QuickSight CloudFormation template for dashboards")
                print("   • Check QuickSight bucket for processed data files")
            else:
                print("\n💥 Bulk load failed. Check Neptune logs for details.")
                
            return success
            
        except Exception as e:
            print(f"❌ Bulk load process failed: {str(e)}")
            return False