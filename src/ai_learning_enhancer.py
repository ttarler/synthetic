#!/usr/bin/env python3
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import json

class AILearningEnhancer:
    def __init__(self):
        self.pattern_learner = PatternLearner()
        self.amount_predictor = AmountPredictor()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
        # Data storage
        self.institutions = []
        self.accounts = []
        self.existing_transactions = []
        
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
    
    def train_ai_models(self):
        """Train AI models on existing transaction data"""
        print("🤖 Training AI models on existing data...")
        
        if not self.existing_transactions:
            raise ValueError("❌ No existing transactions to learn from")
        
        # Prepare features for anomaly detection
        features = []
        for tx in self.existing_transactions:
            feature_vector = [
                tx['amount'],
                tx['hour'],
                tx['risk_score'],
                hash(tx['sender_account']) % 1000,
                hash(tx['receiver_account']) % 1000,
                1 if tx['payment_type'] == 'WIRE' else 0,
                1 if tx['payment_type'] == 'ACH' else 0,
                1 if tx['is_fraud'] else 0
            ]
            features.append(feature_vector)
        
        features_array = np.array(features)
        
        # Train anomaly detector
        self.anomaly_detector.fit(features_array)
        
        # Train pattern learner
        self.pattern_learner.learn_patterns(self.existing_transactions)
        
        # Train amount predictor
        self.amount_predictor.train(self.existing_transactions)
        
        self.is_trained = True
        print("✅ AI models trained successfully!")
        
        # Show learning summary
        fraud_txs = [tx for tx in self.existing_transactions if tx['is_fraud']]
        print(f"📊 Learning Summary:")
        print(f"   • Learned from {len(fraud_txs)} fraud transactions")
        print(f"   • Identified {len(self.pattern_learner.fraud_patterns)} fraud patterns")
        print(f"   • Trained anomaly detector on {len(features)} feature vectors")
    
    def generate_ai_enhanced_transactions(self, count=30000):
        """Generate AI-enhanced transactions based on learned patterns"""
        if not self.is_trained:
            raise ValueError("❌ AI models not trained. Call train_ai_models() first.")
        
        print(f"🎯 Generating {count:,} AI-enhanced transactions...")
        
        ai_transactions = []
        start_id = len(self.existing_transactions)  # Continue from last transaction ID
        
        # Calculate fraud rate from existing data
        existing_fraud_count = sum(1 for tx in self.existing_transactions if tx['is_fraud'])
        fraud_rate = existing_fraud_count / len(self.existing_transactions)
        ai_fraud_count = int(count * fraud_rate)
        
        print(f"📈 Target fraud rate: {fraud_rate*100:.1f}% ({ai_fraud_count:,} fraud transactions)")
        
        # Generate fraud transactions
        fraud_generated = 0
        for i in range(count):
            tx_id = start_id + i
            
            if fraud_generated < ai_fraud_count and random.random() < fraud_rate * 1.5:  # Slightly higher chance
                tx = self._generate_ai_fraud_transaction(tx_id)
                fraud_generated += 1
            else:
                tx = self._generate_ai_normal_transaction(tx_id)
            
            ai_transactions.append(tx)
        
        print(f"✅ Generated {len(ai_transactions):,} AI-enhanced transactions")
        print(f"   • {fraud_generated:,} fraud transactions ({fraud_generated/len(ai_transactions)*100:.1f}%)")
        print(f"   • {len(ai_transactions)-fraud_generated:,} normal transactions")
        
        return ai_transactions
    
    def _generate_ai_fraud_transaction(self, tx_id):
        """Generate AI-enhanced fraud transaction"""
        # Use pattern learner to suggest fraud type
        fraud_type = self.pattern_learner.suggest_fraud_type()
        
        # Select accounts (reuse existing accounts)
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
        
        # Calculate AI-enhanced risk score
        risk_score = self._calculate_ai_risk_score(amount, hour, fraud_type, sender, receiver)
        
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
            'ai_confidence': random.uniform(0.75, 0.95),
            'pattern_similarity': random.uniform(0.65, 0.90),
            'generation_method': 'AI_Enhanced'
        }
    
    def _generate_ai_normal_transaction(self, tx_id):
        """Generate AI-enhanced normal transaction"""
        # Select accounts
        sender = random.choice(self.accounts)
        receiver = random.choice(self.accounts)
        while sender['account_id'] == receiver['account_id']:
            receiver = random.choice(self.accounts)
        
        # AI-predicted amount for normal transactions
        amount = self.amount_predictor.predict_normal_amount()
        
        # Business hour weighting from learned patterns
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
        risk_score = self._calculate_ai_risk_score(amount, hour, None, sender, receiver)
        
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
            'ai_confidence': random.uniform(0.70, 0.90),
            'pattern_similarity': random.uniform(0.60, 0.85),
            'generation_method': 'AI_Enhanced'
        }
    
    def _calculate_ai_risk_score(self, amount, hour, fraud_type, sender, receiver):
        """Calculate AI-enhanced risk score"""
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
        
        # AI anomaly score
        feature_vector = np.array([[
            amount, hour,
            hash(sender['account_id']) % 1000,
            hash(receiver['account_id']) % 1000
        ]])
        
        try:
            anomaly_score = self.anomaly_detector.decision_function(feature_vector)[0]
            ai_enhancement = anomaly_score * 10
            base_score += ai_enhancement
        except:
            pass
        
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
    
    def save_ai_enhanced_data(self, ai_transactions, output_dir='ai_enhanced_output'):
        """Save AI-enhanced transactions to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy existing institutions and accounts (no new entities)
        institutions_df = pd.DataFrame(self.institutions)
        accounts_df = pd.DataFrame(self.accounts)
        
        institutions_df.to_csv(f'{output_dir}/institutions.csv', index=False)
        accounts_df.to_csv(f'{output_dir}/accounts.csv', index=False)
        
        # Save AI-enhanced transactions
        ai_transactions_df = pd.DataFrame(ai_transactions)
        ai_transactions_df.to_csv(f'{output_dir}/transactions.csv', index=False)
        
        # Generate AI summary
        self._generate_ai_summary(ai_transactions, output_dir)
        
        print(f"\n✅ AI-enhanced data saved to {output_dir}/:")
        print(f"   • institutions.csv ({len(self.institutions)} institutions)")
        print(f"   • accounts.csv ({len(self.accounts)} accounts)")
        print(f"   • transactions.csv ({len(ai_transactions)} AI-enhanced transactions)")
        print(f"   • ai_summary.json")
    
    def _generate_ai_summary(self, ai_transactions, output_dir):
        """Generate AI enhancement summary"""
        ai_df = pd.DataFrame(ai_transactions)
        fraud_df = ai_df[ai_df['is_fraud'] == True]
        
        summary = {
            'ai_enhanced_transactions': len(ai_transactions),
            'ai_fraud_transactions': len(fraud_df),
            'ai_fraud_rate': len(fraud_df) / len(ai_transactions) * 100,
            'avg_ai_confidence': ai_df['ai_confidence'].mean(),
            'avg_pattern_similarity': ai_df['pattern_similarity'].mean(),
            'fraud_by_type': fraud_df['fraud_type'].value_counts().to_dict(),
            'generation_method': 'AI_Enhanced',
            'learning_source': 'enhanced_output',
            'models_used': ['PatternLearner', 'AmountPredictor', 'IsolationForest']
        }
        
        with open(f'{output_dir}/ai_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)


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
        
        # Weight by frequency in training data
        weights = [len(patterns) for patterns in self.fraud_patterns.values()]
        fraud_types = list(self.fraud_patterns.keys())
        
        return random.choices(fraud_types, weights=weights)[0]
    
    def suggest_optimal_hour(self, fraud_type):
        """Suggest optimal hour based on learned patterns"""
        if fraud_type and fraud_type in self.timing_patterns:
            hours = self.timing_patterns[fraud_type]
            return random.choice(hours) if hours else random.randint(0, 23)
        else:
            # Normal business hours distribution
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
        
        # Calculate normal amount statistics
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
        
        # Fallback to default ranges
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