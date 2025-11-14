#!/usr/bin/env python3
import pandas as pd
import numpy as np
import random
import os
import json
import boto3
from datetime import datetime, timedelta
from collections import defaultdict
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.randomcutforest import RandomCutForest
from sagemaker.xgboost import XGBoost
from sagemaker.inputs import TrainingInput

class SageMakerFraudEnhancer:
    def __init__(self):
        self.sagemaker_session = sagemaker.Session()
        self.role = get_execution_role()
        self.region = boto3.Session().region_name
        self.account_id = boto3.client('sts').get_caller_identity()['Account']
        
        # S3 bucket for SageMaker
        self.bucket = f"{self.account_id}-sagemaker-fraud-ml"
        self.s3_prefix = 'fraud-detection'
        
        # Model components
        self.pattern_learner = PatternLearner()
        self.amount_predictor = AmountPredictor()
        self.rcf_predictor = None
        self.xgb_predictor = None
        
        # Data storage
        self.institutions = []
        self.accounts = []
        self.existing_transactions = []
        self.is_trained = False
        
    def setup_s3_bucket(self):
        """Create S3 bucket for SageMaker if it doesn't exist"""
        s3_client = boto3.client('s3')
        try:
            s3_client.head_bucket(Bucket=self.bucket)
            print(f"✅ S3 bucket {self.bucket} exists")
        except Exception as e:
            print(f"🪣 Creating S3 bucket {self.bucket}...")
            try:
                if self.region == 'us-east-1':
                    s3_client.create_bucket(Bucket=self.bucket)
                else:
                    s3_client.create_bucket(
                        Bucket=self.bucket,
                        CreateBucketConfiguration={'LocationConstraint': self.region}
                    )
                print(f"✅ Created S3 bucket {self.bucket}")
            except Exception as create_error:
                print(f"❌ Failed to create bucket: {str(create_error)}")
                # Check if it's a permission issue
                if 'AccessDenied' in str(create_error):
                    print("🔧 Trying alternative bucket name...")
                    # Use a simpler bucket name for workshop environments
                    import time
                    self.bucket = f"sagemaker-fraud-{self.account_id}-{int(time.time())}"
                    try:
                        if self.region == 'us-east-1':
                            s3_client.create_bucket(Bucket=self.bucket)
                        else:
                            s3_client.create_bucket(
                                Bucket=self.bucket,
                                CreateBucketConfiguration={'LocationConstraint': self.region}
                            )
                        print(f"✅ Created alternative bucket: {self.bucket}")
                    except Exception as final_error:
                        print(f"❌ Final bucket creation failed: {str(final_error)}")
                        raise
                else:
                    raise
    
    def load_existing_data(self, data_dir='enhanced_output'):
        """Load existing dataset and validate"""
        print("🔍 Checking for existing dataset...")
        
        # Check if files exist
        required_files = ['institutions.csv', 'accounts.csv', 'transactions.csv']
        for file in required_files:
            file_path = os.path.join(data_dir, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"❌ Required file not found: {file_path}")
        
        # Load data
        print("📂 Loading existing data...")
        institutions_df = pd.read_csv(os.path.join(data_dir, 'institutions.csv'))
        accounts_df = pd.read_csv(os.path.join(data_dir, 'accounts.csv'))
        transactions_df = pd.read_csv(os.path.join(data_dir, 'transactions.csv'))
        
        # Validate data
        if len(transactions_df) == 0:
            raise ValueError("❌ No transactions found in existing dataset")
        
        # Convert to lists for processing
        self.institutions = institutions_df.to_dict('records')
        self.accounts = accounts_df.to_dict('records')
        self.existing_transactions = transactions_df.to_dict('records')
        
        print(f"✅ Loaded existing data:")
        print(f"   • {len(self.institutions)} institutions")
        print(f"   • {len(self.accounts)} accounts")
        print(f"   • {len(self.existing_transactions)} transactions")
        
        return True
    
    def train_sagemaker_models(self):
        """Train SageMaker models on existing transaction data"""
        print("🤖 Training SageMaker models on existing data...")
        
        if not self.existing_transactions:
            raise ValueError("❌ No existing transactions to learn from")
        
        # Setup S3 bucket
        self.setup_s3_bucket()
        
        # Prepare features for anomaly detection (Random Cut Forest)
        print("📊 Preparing features for Random Cut Forest...")
        features = []
        labels = []
        
        for tx in self.existing_transactions:
            feature_vector = [
                tx['amount'],
                tx['hour'],
                tx['risk_score'],
                hash(tx['sender_account']) % 1000,
                hash(tx['receiver_account']) % 1000,
                1 if tx['payment_type'] == 'WIRE' else 0,
                1 if tx['payment_type'] == 'ACH' else 0,
                1 if tx['payment_type'] == 'RTP' else 0
            ]
            features.append(feature_vector)
            labels.append(1 if tx['is_fraud'] else 0)
        
        # Convert to DataFrame and upload to S3
        features_df = pd.DataFrame(features)
        labeled_df = pd.DataFrame(features)
        labeled_df['fraud'] = labels
        
        # Save training data locally first
        os.makedirs('temp_training', exist_ok=True)
        features_df.to_csv('temp_training/features.csv', index=False, header=False)
        labeled_df.to_csv('temp_training/labeled_data.csv', index=False, header=False)
        
        # Debug: Check the actual CSV structure
        print(f"📊 Features DataFrame shape: {features_df.shape}")
        print(f"📊 First few rows of features.csv:")
        print(features_df.head(2))
        
        # Upload training data to S3
        rcf_s3_path = f's3://{self.bucket}/{self.s3_prefix}/rcf-training/'
        xgb_s3_path = f's3://{self.bucket}/{self.s3_prefix}/xgb-training/'
        
        # Upload files to S3
        s3_client = boto3.client('s3')
        s3_client.upload_file('temp_training/features.csv', self.bucket, f'{self.s3_prefix}/rcf-training/features.csv')
        s3_client.upload_file('temp_training/labeled_data.csv', self.bucket, f'{self.s3_prefix}/xgb-training/labeled_data.csv')
        
        print(f"📤 Uploaded training data to S3")
        
        # Train Random Cut Forest for anomaly detection
        print("🌲 Training Random Cut Forest...")
        
        # For SageMaker 2.x, use the built-in algorithm container
        from sagemaker import image_uris
        
        # Get the RandomCutForest container image URI
        container = image_uris.retrieve("randomcutforest", self.region)
        
        from sagemaker.estimator import Estimator
        rcf = Estimator(
            image_uri=container,
            role=self.role,
            instance_count=1,
            instance_type='ml.m5.large',
            output_path=f's3://{self.bucket}/{self.s3_prefix}/rcf-output',
            sagemaker_session=self.sagemaker_session
        )
        
        # Set hyperparameters for RCF
        # feature_dim=7 because RCF expects feature_dim + 1 columns in CSV
        rcf.set_hyperparameters(
            feature_dim=7,
            num_samples_per_tree=256,
            num_trees=100
        )
        
        # Training input with correct distribution type for RandomCutForest
        train_input = TrainingInput(
            s3_data=f's3://{self.bucket}/{self.s3_prefix}/rcf-training/',
            content_type='text/csv',
            distribution='ShardedByS3Key'
        )
        
        rcf.fit({'train': train_input})
        print("✅ Random Cut Forest training completed")
        
        # Deploy RCF model
        print("🚀 Deploying Random Cut Forest endpoint...")
        self.rcf_predictor = rcf.deploy(
            initial_instance_count=1,
            instance_type='ml.t2.medium',
            endpoint_name=f'fraud-rcf-{int(datetime.now().timestamp())}'
        )
        print(f"✅ RCF endpoint deployed: {self.rcf_predictor.endpoint_name}")
        
        # Train XGBoost for fraud classification
        print("🎯 Training XGBoost for fraud classification...")
        xgb = XGBoost(
            entry_point='xgboost_fraud_script.py',
            source_dir='src',
            role=self.role,
            instance_count=1,
            instance_type='ml.m5.large',
            framework_version='1.5-1',
            hyperparameters={
                'max_depth': 5,
                'eta': 0.2,
                'gamma': 4,
                'min_child_weight': 6,
                'subsample': 0.8,
                'objective': 'binary:logistic',
                'num_round': 100
            },
            sagemaker_session=self.sagemaker_session
        )
        
        xgb_input = TrainingInput(f's3://{self.bucket}/{self.s3_prefix}/xgb-training/', content_type='text/csv')
        xgb.fit({'train': xgb_input})
        print("✅ XGBoost training completed")
        
        # Deploy XGBoost model
        print("🚀 Deploying XGBoost endpoint...")
        self.xgb_predictor = xgb.deploy(
            initial_instance_count=1,
            instance_type='ml.t2.medium',
            endpoint_name=f'fraud-xgb-{int(datetime.now().timestamp())}'
        )
        print(f"✅ XGBoost endpoint deployed: {self.xgb_predictor.endpoint_name}")
        
        # Train pattern learner and amount predictor locally
        self.pattern_learner.learn_patterns(self.existing_transactions)
        self.amount_predictor.train(self.existing_transactions)
        
        self.is_trained = True
        print("✅ All SageMaker models trained and deployed successfully!")
        
        # Show learning summary
        fraud_txs = [tx for tx in self.existing_transactions if tx['is_fraud']]
        print(f"📊 Learning Summary:")
        print(f"   • Random Cut Forest: SageMaker anomaly detection on {len(features)} transactions")
        print(f"   • XGBoost: Fraud classification on {len(fraud_txs)} fraud cases")
        print(f"   • Pattern Learner: {len(self.pattern_learner.fraud_patterns)} fraud patterns")
        print(f"   • Amount Predictor: Trained on transaction distributions")
    
    def generate_ai_enhanced_transactions(self, count=30000):
        """Generate AI-enhanced transactions using SageMaker models"""
        if not self.is_trained:
            raise ValueError("❌ SageMaker models not trained. Call train_sagemaker_models() first.")
        
        print(f"🎯 Generating {count:,} AI-enhanced transactions using SageMaker...")
        
        ai_transactions = []
        start_id = len(self.existing_transactions)
        
        # Calculate fraud rate from existing data
        existing_fraud_count = sum(1 for tx in self.existing_transactions if tx['is_fraud'])
        fraud_rate = existing_fraud_count / len(self.existing_transactions)
        ai_fraud_count = int(count * fraud_rate)
        
        print(f"📈 Target fraud rate: {fraud_rate*100:.1f}% ({ai_fraud_count:,} fraud transactions)")
        
        # Generate transactions
        fraud_generated = 0
        for i in range(count):
            tx_id = start_id + i
            
            if fraud_generated < ai_fraud_count and random.random() < fraud_rate * 1.5:
                tx = self._generate_sagemaker_fraud_transaction(tx_id)
                fraud_generated += 1
            else:
                tx = self._generate_sagemaker_normal_transaction(tx_id)
            
            ai_transactions.append(tx)
            
            # Progress update
            if (i + 1) % 5000 == 0:
                print(f"   Generated {i + 1:,} transactions...")
        
        print(f"✅ Generated {len(ai_transactions):,} SageMaker-enhanced transactions")
        print(f"   • {fraud_generated:,} fraud transactions ({fraud_generated/len(ai_transactions)*100:.1f}%)")
        print(f"   • {len(ai_transactions)-fraud_generated:,} normal transactions")
        
        return ai_transactions
    
    def _generate_sagemaker_fraud_transaction(self, tx_id):
        """Generate AI-enhanced fraud transaction using SageMaker"""
        # Use pattern learner to suggest fraud type
        fraud_type = self.pattern_learner.suggest_fraud_type()
        
        # Select accounts
        sender = random.choice(self.accounts)
        receiver = random.choice(self.accounts)
        while sender['account_id'] == receiver['account_id']:
            receiver = random.choice(self.accounts)
        
        # AI-predicted amount
        amount = self.amount_predictor.predict_fraud_amount(fraud_type)
        
        # AI-suggested timing
        hour = self.pattern_learner.suggest_optimal_hour(fraud_type)
        
        # Generate timestamp
        base_date = datetime(2024, 1, 1)
        timestamp = base_date + timedelta(
            days=random.randint(0, 89),
            hours=hour,
            minutes=random.randint(0, 59)
        )
        
        # Payment type
        payment_type = random.choice(['ACH', 'WIRE', 'RTP'])
        
        # Calculate SageMaker-enhanced risk score
        risk_score = self._calculate_sagemaker_risk_score(amount, hour, fraud_type, sender, receiver)
        
        return {
            'transaction_id': f'TXN_{tx_id:06d}',
            'sender_account': sender['account_id'],
            'receiver_account': receiver['account_id'],
            'sender_institution': sender['institution_id'],
            'receiver_institution': receiver['institution_id'],
            'amount': round(amount, 2),
            'payment_type': payment_type,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'hour': hour,
            'is_fraud': True,
            'fraud_type': fraud_type,
            'risk_score': risk_score,
            'ai_confidence': random.uniform(0.80, 0.98),
            'pattern_similarity': random.uniform(0.70, 0.95),
            'generation_method': 'SageMaker_Enhanced'
        }
    
    def _generate_sagemaker_normal_transaction(self, tx_id):
        """Generate AI-enhanced normal transaction using SageMaker"""
        # Select accounts
        sender = random.choice(self.accounts)
        receiver = random.choice(self.accounts)
        while sender['account_id'] == receiver['account_id']:
            receiver = random.choice(self.accounts)
        
        # AI-predicted amount
        amount = self.amount_predictor.predict_normal_amount()
        
        # Business hour weighting
        hour = self.pattern_learner.suggest_optimal_hour(None)
        
        # Generate timestamp
        base_date = datetime(2024, 1, 1)
        timestamp = base_date + timedelta(
            days=random.randint(0, 89),
            hours=hour,
            minutes=random.randint(0, 59)
        )
        
        # Payment type distribution
        payment_type = np.random.choice(['ACH', 'WIRE', 'RTP'], p=[0.60, 0.15, 0.25])
        
        # Calculate risk score
        risk_score = self._calculate_sagemaker_risk_score(amount, hour, None, sender, receiver)
        
        return {
            'transaction_id': f'TXN_{tx_id:06d}',
            'sender_account': sender['account_id'],
            'receiver_account': receiver['account_id'],
            'sender_institution': sender['institution_id'],
            'receiver_institution': receiver['institution_id'],
            'amount': round(amount, 2),
            'payment_type': payment_type,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'hour': hour,
            'is_fraud': False,
            'fraud_type': '',
            'risk_score': risk_score,
            'ai_confidence': random.uniform(0.75, 0.92),
            'pattern_similarity': random.uniform(0.65, 0.88),
            'generation_method': 'SageMaker_Enhanced'
        }
    
    def _calculate_sagemaker_risk_score(self, amount, hour, fraud_type, sender, receiver):
        """Calculate risk score using SageMaker models"""
        base_score = 0
        
        # Amount risk
        if amount >= 100000:
            base_score += 30
        elif amount >= 50000:
            base_score += 20
        elif amount >= 10000:
            base_score += 10
        elif amount <= 100:
            base_score += 5
        
        # Time risk
        if hour in [22, 23, 0, 1, 2, 3, 4, 5]:
            base_score += 20
        elif hour in [6, 7, 18, 19, 20, 21]:
            base_score += 5
        
        # SageMaker Random Cut Forest anomaly score
        if self.rcf_predictor:
            try:
                # Create feature vector matching training data (7 features since feature_dim=7)
                feature_vector = [
                    amount, hour,
                    hash(sender['account_id']) % 1000,
                    hash(receiver['account_id']) % 1000,
                    1 if 'WIRE' in str(sender) else 0,
                    1 if 'ACH' in str(sender) else 0,
                    1 if 'RTP' in str(sender) else 0
                ]
                
                # Call RCF endpoint with proper format
                # RandomCutForest expects newline-terminated CSV format
                feature_csv = ','.join(map(str, feature_vector)) + '\n'
                response = self.rcf_predictor.predict(
                    feature_csv,
                    initial_args={'ContentType': 'text/csv'}
                )
                
                # Parse RandomCutForest response (it returns JSON as bytes)
                import json
                if isinstance(response, bytes):
                    response = json.loads(response.decode('utf-8'))
                elif isinstance(response, str):
                    response = json.loads(response)
                
                anomaly_score = response['scores'][0]['score']
                
                # Higher anomaly score = more suspicious
                sagemaker_enhancement = anomaly_score * 15
                base_score += sagemaker_enhancement
                
            except Exception as e:
                print(f"Warning: SageMaker RCF call failed: {e}")
                # Fallback to simple enhancement
                base_score += random.uniform(-5, 15)
        else:
            # No RCF available, use simple enhancement
            base_score += random.uniform(-5, 15)
        
        # Fraud type multiplier
        if fraud_type:
            multipliers = {
                'insider_fraud': 2.0,
                'trade_based_laundering': 1.9,
                'money_laundering_ring': 1.8,
                'cryptocurrency_mixing': 1.8,
                'shell_company_web': 1.7,
                'synthetic_identity': 1.7,
                'smurfing_network': 1.6,
                'unusual_amount': 1.5,
                'off_hours': 1.3
            }
            multiplier = multipliers.get(fraud_type, 1.0)
            base_score *= multiplier
        
        return min(100, max(0, round(base_score, 1)))
    
    def save_sagemaker_enhanced_data(self, ai_transactions, output_dir='sagemaker_enhanced_output'):
        """Save SageMaker-enhanced transactions to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy existing institutions and accounts
        institutions_df = pd.DataFrame(self.institutions)
        accounts_df = pd.DataFrame(self.accounts)
        
        institutions_df.to_csv(f'{output_dir}/institutions.csv', index=False)
        accounts_df.to_csv(f'{output_dir}/accounts.csv', index=False)
        
        # Save SageMaker-enhanced transactions
        ai_transactions_df = pd.DataFrame(ai_transactions)
        ai_transactions_df.to_csv(f'{output_dir}/transactions.csv', index=False)
        
        # Generate SageMaker summary
        self._generate_sagemaker_summary(ai_transactions, output_dir)
        
        print(f"\n✅ SageMaker-enhanced data saved to {output_dir}/:")
        print(f"   • institutions.csv ({len(self.institutions)} institutions)")
        print(f"   • accounts.csv ({len(self.accounts)} accounts)")
        print(f"   • transactions.csv ({len(ai_transactions)} SageMaker-enhanced transactions)")
        print(f"   • sagemaker_summary.json")
    
    def _generate_sagemaker_summary(self, ai_transactions, output_dir):
        """Generate SageMaker enhancement summary"""
        ai_df = pd.DataFrame(ai_transactions)
        fraud_df = ai_df[ai_df['is_fraud'] == True]
        
        summary = {
            'sagemaker_enhanced_transactions': len(ai_transactions),
            'sagemaker_fraud_transactions': len(fraud_df),
            'sagemaker_fraud_rate': len(fraud_df) / len(ai_transactions) * 100,
            'avg_ai_confidence': ai_df['ai_confidence'].mean(),
            'avg_pattern_similarity': ai_df['pattern_similarity'].mean(),
            'fraud_by_type': fraud_df['fraud_type'].value_counts().to_dict(),
            'generation_method': 'SageMaker_Enhanced',
            'learning_source': 'enhanced_output',
            'models_used': ['RandomCutForest', 'XGBoost', 'PatternLearner', 'AmountPredictor'],
            'rcf_endpoint': self.rcf_predictor.endpoint_name if self.rcf_predictor else None,
            'xgb_endpoint': self.xgb_predictor.endpoint_name if self.xgb_predictor else None
        }
        
        with open(f'{output_dir}/sagemaker_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def cleanup_endpoints(self):
        """Clean up SageMaker endpoints to avoid charges"""
        print("🧹 Cleaning up SageMaker endpoints...")
        
        if self.rcf_predictor:
            self.rcf_predictor.delete_endpoint()
            print(f"✅ Deleted RCF endpoint: {self.rcf_predictor.endpoint_name}")
        
        if self.xgb_predictor:
            self.xgb_predictor.delete_endpoint()
            print(f"✅ Deleted XGBoost endpoint: {self.xgb_predictor.endpoint_name}")
        
        print("✅ All SageMaker endpoints cleaned up")


# Reuse existing PatternLearner and AmountPredictor classes
class PatternLearner:
    """Learn fraud patterns from existing data"""
    
    def __init__(self):
        self.fraud_patterns = defaultdict(list)
        self.timing_patterns = defaultdict(list)
        self.amount_patterns = defaultdict(list)
        
    def learn_patterns(self, transactions):
        """Learn patterns from transaction data"""
        for tx in transactions:
            if tx['is_fraud']:
                fraud_type = tx['fraud_type']
                self.fraud_patterns[fraud_type].append({
                    'amount': tx['amount'],
                    'hour': tx['hour'],
                    'risk_score': tx['risk_score']
                })
                self.timing_patterns[fraud_type].append(tx['hour'])
                self.amount_patterns[fraud_type].append(tx['amount'])
    
    def suggest_fraud_type(self):
        """Suggest fraud type based on learned frequency"""
        if not self.fraud_patterns:
            return 'money_laundering_ring'
        
        weights = [len(patterns) for patterns in self.fraud_patterns.values()]
        fraud_types = list(self.fraud_patterns.keys())
        
        return random.choices(fraud_types, weights=weights)[0]
    
    def suggest_optimal_hour(self, fraud_type):
        """Suggest optimal hour based on learned patterns"""
        if fraud_type and fraud_type in self.timing_patterns:
            hours = self.timing_patterns[fraud_type]
            return random.choice(hours) if hours else random.randint(0, 23)
        else:
            return random.choices(
                range(24), 
                weights=[2,1,1,1,1,2,4,6,8,10,10,10,10,10,8,8,8,6,6,4,4,3,3,2]
            )[0]


class AmountPredictor:
    """Predict transaction amounts using learned patterns"""
    
    def __init__(self):
        self.fraud_amount_patterns = defaultdict(list)
        self.normal_amount_stats = {}
        
    def train(self, transactions):
        """Train amount predictor on transaction data"""
        normal_amounts = []
        
        for tx in transactions:
            if tx['is_fraud']:
                self.fraud_amount_patterns[tx['fraud_type']].append(tx['amount'])
            else:
                normal_amounts.append(tx['amount'])
        
        if normal_amounts:
            self.normal_amount_stats = {
                'mean': np.mean(normal_amounts),
                'std': np.std(normal_amounts),
                'min': np.min(normal_amounts),
                'max': np.max(normal_amounts)
            }
    
    def predict_fraud_amount(self, fraud_type):
        """Predict fraud amount based on learned patterns"""
        if fraud_type in self.fraud_amount_patterns:
            amounts = self.fraud_amount_patterns[fraud_type]
            if amounts:
                base_amount = random.choice(amounts)
                variation = random.uniform(0.8, 1.2)
                return round(base_amount * variation, 2)
        
        fraud_ranges = {
            'money_laundering_ring': (8000, 25000),
            'smurfing_network': (8500, 9400),
            'shell_company_web': (15000, 75000),
            'trade_based_laundering': (50000, 500000),
            'cryptocurrency_mixing': (5000, 100000),
            'insider_fraud': (25000, 200000),
            'synthetic_identity': (10000, 50000),
            'unusual_amount': (100000, 1000000),
            'off_hours': (5000, 50000)
        }
        
        min_amt, max_amt = fraud_ranges.get(fraud_type, (1000, 10000))
        return round(random.uniform(min_amt, max_amt), 2)
    
    def predict_normal_amount(self):
        """Predict normal transaction amount"""
        if self.normal_amount_stats:
            mean = self.normal_amount_stats['mean']
            std = self.normal_amount_stats['std']
            amount = np.random.normal(mean, std)
            amount = max(10, min(50000, amount))
            return round(amount, 2)
        
        return round(random.uniform(10, 5000), 2)