import pandas as pd
import numpy as np
import yaml
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import json

class EnhancedFraudGenerator:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.institutions = []
        self.accounts = []
        self.transactions = []
        self.fraud_rings = {}
        self.shell_accounts = set()
        self.synthetic_accounts = set()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def generate_network(self) -> Dict[str, pd.DataFrame]:
        """Generate complete enhanced fraud network"""
        print("Generating enhanced fraud network...")
        
        self._generate_institutions()
        self._generate_accounts()
        self._create_fraud_rings()
        self._generate_all_transactions()
        
        return {
            'institutions': pd.DataFrame(self.institutions),
            'accounts': pd.DataFrame(self.accounts),
            'transactions': pd.DataFrame(self.transactions)
        }
    
    def _generate_institutions(self):
        """Generate institutions based on config"""
        inst_config = self.config['network_structure']['institutions']
        count = inst_config['count']
        types = inst_config['types']
        
        for i in range(count):
            # Select institution type based on probability
            inst_type = np.random.choice(
                [t['type'] for t in types],
                p=[t['probability'] for t in types]
            )
            
            # Get type config
            type_config = next(t for t in types if t['type'] == inst_type)
            
            institution = {
                'institution_id': f'INST_{i:03d}',
                'name': f'Financial Institution {i+1}',
                'type': inst_type,
                'channels': ','.join(type_config['channels'])
            }
            self.institutions.append(institution)
    
    def _generate_accounts(self):
        """Generate accounts with special account types"""
        acc_config = self.config['network_structure']['accounts']
        total_count = acc_config['total_count']
        balance_range = acc_config['balance_range']
        account_types = acc_config['types']
        
        # Special account percentages
        shell_pct = acc_config['special_accounts']['shell_accounts']['percentage']
        synthetic_pct = acc_config['special_accounts']['synthetic_accounts']['percentage']
        
        for i in range(total_count):
            account_id = f'ACC_{i:06d}'
            
            # Determine if special account
            rand_val = random.random()
            if rand_val < shell_pct:
                # Shell account
                balance_range_special = acc_config['special_accounts']['shell_accounts']['balance_range']
                balance = random.uniform(*balance_range_special)
                self.shell_accounts.add(account_id)
                is_shell, is_synthetic = True, False
            elif rand_val < shell_pct + synthetic_pct:
                # Synthetic account
                balance_range_special = acc_config['special_accounts']['synthetic_accounts']['balance_range']
                balance = random.uniform(*balance_range_special)
                self.synthetic_accounts.add(account_id)
                is_shell, is_synthetic = False, True
            else:
                # Normal account
                balance = random.uniform(*balance_range)
                is_shell, is_synthetic = False, False
            
            account = {
                'account_id': account_id,
                'institution_id': random.choice(self.institutions)['institution_id'],
                'account_type': random.choice(account_types),
                'balance': round(balance, 2),
                'is_shell': is_shell,
                'is_synthetic': is_synthetic
            }
            self.accounts.append(account)
    
    def _create_fraud_rings(self):
        """Create money laundering rings based on config"""
        fraud_config = self.config['fraud_patterns']['fraud_types']['money_laundering_ring']
        total_fraud = int(self.config['transaction_generation']['total_transactions'] * 
                         self.config['fraud_patterns']['overall_fraud_rate'])
        
        ring_transactions = int(total_fraud * fraud_config['percentage_of_fraud'])
        transactions_per_ring = fraud_config['characteristics']['transactions_per_ring']
        ring_count = ring_transactions // transactions_per_ring
        
        ring_size_range = fraud_config['characteristics']['ring_size_range']
        
        for ring_id in range(ring_count):
            ring_size = random.randint(*ring_size_range)
            ring_accounts = random.sample([acc['account_id'] for acc in self.accounts], ring_size)
            self.fraud_rings[f'RING_{ring_id:03d}'] = ring_accounts
    
    def _generate_all_transactions(self):
        """Generate all transactions (normal + fraud)"""
        tx_config = self.config['transaction_generation']
        total_transactions = tx_config['total_transactions']
        fraud_rate = self.config['fraud_patterns']['overall_fraud_rate']
        
        # Calculate fraud transaction counts
        fraud_counts = self._calculate_fraud_counts(total_transactions, fraud_rate)
        
        # Generate fraud transactions
        tx_id = 0
        fraud_transactions = []
        
        for fraud_type, count in fraud_counts.items():
            print(f"Generating {count} {fraud_type} transactions...")
            for _ in range(count):
                fraud_tx = self._generate_fraud_transaction(fraud_type, tx_id)
                fraud_transactions.append(fraud_tx)
                tx_id += 1
        
        # Generate normal transactions
        normal_count = total_transactions - len(fraud_transactions)
        print(f"Generating {normal_count} normal transactions...")
        
        normal_transactions = []
        for _ in range(normal_count):
            normal_tx = self._generate_normal_transaction(tx_id)
            normal_transactions.append(normal_tx)
            tx_id += 1
        
        # Combine and shuffle
        self.transactions = fraud_transactions + normal_transactions
        random.shuffle(self.transactions)
        
        print(f"Generated {len(self.transactions)} total transactions")
        print(f"Fraud rate: {len(fraud_transactions)/len(self.transactions)*100:.1f}%")
    
    def _calculate_fraud_counts(self, total_transactions: int, fraud_rate: float) -> Dict[str, int]:
        """Calculate number of transactions for each fraud type"""
        total_fraud = int(total_transactions * fraud_rate)
        fraud_types = self.config['fraud_patterns']['fraud_types']
        
        fraud_counts = {}
        for fraud_type, config in fraud_types.items():
            count = int(total_fraud * config['percentage_of_fraud'])
            fraud_counts[fraud_type] = count
            
        return fraud_counts
    
    def _generate_fraud_transaction(self, fraud_type: str, tx_id: int) -> Dict[str, Any]:
        """Generate specific fraud transaction based on type"""
        fraud_config = self.config['fraud_patterns']['fraud_types'][fraud_type]
        characteristics = fraud_config['characteristics']
        
        # Select accounts based on fraud type
        sender, receiver = self._select_fraud_accounts(fraud_type, characteristics)
        
        # Generate amount based on fraud type
        amount = random.uniform(*characteristics['amount_range'])
        
        # Generate timestamp
        timestamp = self._generate_fraud_timestamp(fraud_type, characteristics)
        
        # Select payment type
        payment_type = random.choice(['ACH', 'WIRE', 'RTP'])
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(amount, timestamp.hour, fraud_type, sender, receiver)
        
        return {
            'transaction_id': f'TXN_{tx_id:06d}',
            'sender_account': sender['account_id'],
            'receiver_account': receiver['account_id'],
            'sender_institution': sender['institution_id'],
            'receiver_institution': receiver['institution_id'],
            'amount': round(amount, 2),
            'payment_type': payment_type,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'hour': timestamp.hour,
            'is_fraud': True,
            'fraud_type': fraud_type,
            'risk_score': risk_score
        }
    
    def _select_fraud_accounts(self, fraud_type: str, characteristics: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """Select sender and receiver accounts based on fraud type"""
        if fraud_type == 'money_laundering_ring' and self.fraud_rings:
            ring_id = random.choice(list(self.fraud_rings.keys()))
            ring_accounts = self.fraud_rings[ring_id]
            sender_id, receiver_id = random.sample(ring_accounts, 2)
            sender = next(acc for acc in self.accounts if acc['account_id'] == sender_id)
            receiver = next(acc for acc in self.accounts if acc['account_id'] == receiver_id)
            
        elif fraud_type == 'shell_company_web' and characteristics.get('uses_shell_accounts') and self.shell_accounts:
            sender_id = random.choice(list(self.shell_accounts))
            sender = next(acc for acc in self.accounts if acc['account_id'] == sender_id)
            receiver = random.choice(self.accounts)
            
        elif fraud_type == 'synthetic_identity' and characteristics.get('uses_synthetic_accounts') and self.synthetic_accounts:
            sender_id = random.choice(list(self.synthetic_accounts))
            sender = next(acc for acc in self.accounts if acc['account_id'] == sender_id)
            receiver = random.choice(self.accounts)
            
        else:
            sender = random.choice(self.accounts)
            receiver = random.choice(self.accounts)
            while sender['account_id'] == receiver['account_id']:
                receiver = random.choice(self.accounts)
        
        return sender, receiver
    
    def _generate_fraud_timestamp(self, fraud_type: str, characteristics: Dict[str, Any]) -> datetime:
        """Generate timestamp based on fraud type characteristics"""
        base_config = self.config['transaction_generation']
        base_date = datetime.strptime(base_config['base_date'], '%Y-%m-%d')
        days_range = base_config['time_period_days']
        
        # Random day within period
        random_day = random.randint(0, days_range - 1)
        
        # Hour selection based on fraud type
        if fraud_type == 'off_hours' or characteristics.get('after_hours_activity'):
            time_windows = characteristics.get('time_windows', [[22, 6]])
            window = random.choice(time_windows)
            if window[0] > window[1]:  # Crosses midnight
                hour = random.choice(list(range(window[0], 24)) + list(range(0, window[1] + 1)))
            else:
                hour = random.randint(window[0], window[1])
        else:
            hour = random.randint(6, 21)  # Business hours
        
        timestamp = base_date + timedelta(
            days=random_day,
            hours=hour,
            minutes=random.randint(0, 59)
        )
        
        return timestamp
    
    def _generate_normal_transaction(self, tx_id: int) -> Dict[str, Any]:
        """Generate normal transaction"""
        tx_config = self.config['transaction_generation']['normal_transactions']
        
        # Select random accounts
        sender = random.choice(self.accounts)
        receiver = random.choice(self.accounts)
        while sender['account_id'] == receiver['account_id']:
            receiver = random.choice(self.accounts)
        
        # Generate amount
        amount = random.uniform(*tx_config['amount_range'])
        
        # Generate timestamp with business hour weighting
        hour_weights = tx_config['hour_weights']
        hour = random.choices(range(24), weights=hour_weights)[0]
        
        base_date = datetime.strptime(self.config['transaction_generation']['base_date'], '%Y-%m-%d')
        timestamp = base_date + timedelta(
            days=random.randint(0, self.config['transaction_generation']['time_period_days'] - 1),
            hours=hour,
            minutes=random.randint(0, 59)
        )
        
        # Select payment type
        payment_dist = tx_config['payment_type_distribution']
        payment_type = np.random.choice(
            list(payment_dist.keys()),
            p=list(payment_dist.values())
        )
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(amount, hour, None, sender, receiver)
        
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
            'risk_score': risk_score
        }
    
    def _calculate_risk_score(self, amount: float, hour: int, fraud_type: str, 
                            sender: Dict, receiver: Dict) -> float:
        """Calculate risk score based on config rules"""
        risk_config = self.config['risk_scoring']
        score = 0
        
        # Amount risk
        for threshold in risk_config['factors']['amount_risk']['thresholds']:
            if amount >= threshold['amount']:
                score += threshold['points']
                break
        
        # Time risk
        time_factors = risk_config['factors']['time_risk']
        if hour in time_factors['off_hours']:
            score += time_factors['off_hours_points']
        elif hour in time_factors['extended_hours']:
            score += time_factors['extended_hours_points']
        
        # Account risk
        account_factors = risk_config['factors']['account_risk']
        if sender['account_id'] in self.shell_accounts:
            score += account_factors['shell_sender_points']
        if receiver['account_id'] in self.shell_accounts:
            score += account_factors['shell_receiver_points']
        if sender['account_id'] in self.synthetic_accounts:
            score += account_factors['synthetic_sender_points']
        if receiver['account_id'] in self.synthetic_accounts:
            score += account_factors['synthetic_receiver_points']
        
        # Cross-institution risk
        if sender['institution_id'] != receiver['institution_id']:
            cross_inst_points = risk_config['factors']['cross_institution_risk']['different_institution_points']
            score += cross_inst_points
        
        # Fraud type multiplier
        if fraud_type:
            multipliers = risk_config['factors']['fraud_type_multipliers']
            multiplier = multipliers.get(fraud_type, 1.0)
            score *= multiplier
        
        return min(risk_config['max_score'], round(score, 1))
    
    def save_data(self, output_dir: str = None):
        """Save generated data to files"""
        if output_dir is None:
            output_dir = self.config['output_settings']['output_directory']
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save institutions
        institutions_df = pd.DataFrame(self.institutions)
        institutions_df.to_csv(f'{output_dir}/institutions.csv', index=False)
        
        # Save accounts
        accounts_df = pd.DataFrame(self.accounts)
        accounts_df.to_csv(f'{output_dir}/accounts.csv', index=False)
        
        # Save transactions
        transactions_df = pd.DataFrame(self.transactions)
        transactions_df.to_csv(f'{output_dir}/transactions.csv', index=False)
        
        # Generate fraud summary
        self._generate_fraud_summary(output_dir)
        
        print(f"\nFiles saved to {output_dir}/:")
        print(f"- institutions.csv ({len(self.institutions)} institutions)")
        print(f"- accounts.csv ({len(self.accounts)} accounts)")
        print(f"- transactions.csv ({len(self.transactions)} transactions)")
        print(f"- fraud_summary.json")
    
    def _generate_fraud_summary(self, output_dir: str):
        """Generate fraud statistics summary"""
        transactions_df = pd.DataFrame(self.transactions)
        fraud_df = transactions_df[transactions_df['is_fraud'] == True]
        
        summary = {
            'total_transactions': len(self.transactions),
            'fraud_transactions': len(fraud_df),
            'fraud_rate': len(fraud_df) / len(self.transactions) * 100,
            'total_fraud_amount': fraud_df['amount'].sum(),
            'avg_fraud_amount': fraud_df['amount'].mean(),
            'fraud_by_type': fraud_df['fraud_type'].value_counts().to_dict(),
            'fraud_by_payment_type': fraud_df['payment_type'].value_counts().to_dict(),
            'avg_risk_score': transactions_df['risk_score'].mean(),
            'high_risk_transactions': len(transactions_df[transactions_df['risk_score'] >= 70])
        }
        
        with open(f'{output_dir}/fraud_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)