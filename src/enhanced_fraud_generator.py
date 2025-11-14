import pandas as pd
import numpy as np
import yaml
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import json
from src.account_profiles import (
    AccountBehaviorProfile,
    AccountProfileGenerator,
    RelationshipNetwork,
    FrequencyTier,
    RelationshipType
)
from src.schema_mapper import PaymentSchemaMapper
from src.institutional_relationships import InstitutionalRelationships

class EnhancedFraudGenerator:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.institutions = []
        self.accounts = []
        self.transactions = []
        self.fraud_rings = {}
        self.shell_accounts = set()
        self.synthetic_accounts = set()

        # NEW: Realistic transaction generation components
        self.account_profiles: Dict[str, AccountBehaviorProfile] = {}
        self.relationship_network = RelationshipNetwork()
        self.use_realistic_generation = self.config.get('transaction_generation', {}).get('normal_transactions', {}).get('use_realistic_generation', False)

        # Transaction scheduling for recurring patterns
        self.recurring_transactions: List[Dict[str, Any]] = []
        self.transaction_chains: List[List[Dict[str, Any]]] = []

        # Schema mapper for full payment system schema
        self.schema_mapper: Optional[PaymentSchemaMapper] = None

        # Institutional relationships matrix for realistic ODFI-RDFI patterns
        self.institutional_relationships: Optional[InstitutionalRelationships] = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def generate_network(self) -> Dict[str, pd.DataFrame]:
        """Generate complete enhanced fraud network"""
        print("Generating enhanced fraud network...")

        self._generate_institutions()

        # Initialize institutional relationships matrix BEFORE schema mapper
        # This ensures routing numbers from matrix are used
        inst_rel_config = self.config.get('institutional_relationships', {})
        matrix_path = inst_rel_config.get('matrix_path', 'config/odfi_rdfi_matrix.csv')
        create_sample = inst_rel_config.get('create_sample_if_missing', True)

        if matrix_path:
            import os
            if not os.path.exists(matrix_path) and create_sample:
                # Create sample matrix for testing
                print(f"   📝 Matrix file not found, creating sample matrix...")
                sample_gen = InstitutionalRelationships()
                sample_gen.create_sample_matrix(matrix_path, num_institutions=len(self.institutions))

            self.institutional_relationships = InstitutionalRelationships(matrix_path)
            if self.institutional_relationships.is_loaded:
                # Map routing numbers from matrix to generated institutions
                # This must happen BEFORE schema mapper so it uses the correct routing numbers
                self._map_matrix_to_institutions()
        else:
            print("   ℹ️  No institutional relationship matrix configured")

        # Initialize schema mapper after institutions and routing numbers are finalized
        print("🗺️  Initializing payment schema mapper...")
        self.schema_mapper = PaymentSchemaMapper(self.institutions)

        self._generate_accounts()

        # NEW: Generate account behavior profiles and relationship network
        if self.use_realistic_generation:
            print("🎯 Generating realistic account behavior profiles...")
            self._generate_account_profiles()
            print("🕸️  Building account relationship network...")
            self.relationship_network.build_network(self.accounts, self.account_profiles)

        self._create_fraud_rings()
        self._generate_all_transactions()

        # Map all transactions to full payment schema
        print("📋 Mapping transactions to full payment system schema...")
        self._map_transactions_to_schema()

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

    def _generate_account_profiles(self):
        """Generate realistic behavior profiles for each account"""
        for account in self.accounts:
            profile = AccountProfileGenerator.generate_profile(
                account['account_id'],
                account['account_type']
            )
            self.account_profiles[account['account_id']] = profile

        print(f"   Generated {len(self.account_profiles)} account behavior profiles")

        # Print frequency tier distribution
        tier_counts = defaultdict(int)
        for profile in self.account_profiles.values():
            tier_counts[profile.frequency_tier.value] += 1

        print("   Frequency tier distribution:")
        for tier, count in sorted(tier_counts.items()):
            pct = (count / len(self.account_profiles)) * 100
            print(f"     {tier}: {count} ({pct:.1f}%)")

    def _map_matrix_to_institutions(self):
        """
        Map routing numbers from institutional relationship matrix to generated institutions

        Creates a bidirectional mapping so we can use real-world transaction patterns
        """
        if not self.institutional_relationships or not self.institutional_relationships.is_loaded:
            return

        # Get routing numbers from matrix
        matrix_routings = list(self.institutional_relationships.all_institutions)

        # If we have more institutions in matrix than generated, sample subset
        if len(matrix_routings) > len(self.institutions):
            # Prioritize institutions with highest relationship counts
            matrix_stats = self.institutional_relationships.matrix_df.groupby('odfi')['transaction_count'].sum()
            top_routings = matrix_stats.nlargest(len(self.institutions)).index.tolist()
            matrix_routings = top_routings

        # Create deterministic mapping: matrix routing -> generated institution        # Use modulo to ensure consistent mapping
        self.routing_to_institution = {}
        for idx, routing in enumerate(matrix_routings):
            if idx < len(self.institutions):
                inst = self.institutions[idx]
                # Update institution with real routing number from matrix
                original_routing = inst.get('routing_number')
                inst['routing_number'] = routing
                inst['matrix_routing'] = routing  # Keep track that this came from matrix
                self.routing_to_institution[routing] = inst
            else:
                # Hash-based assignment for remaining routings
                hash_idx = int(routing) % len(self.institutions) if routing.isdigit() else hash(routing) % len(self.institutions)
                inst = self.institutions[hash_idx]
                self.routing_to_institution[routing] = inst

        print(f"   🔗 Mapped {len(self.routing_to_institution)} routing numbers to {len(self.institutions)} institutions")

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
        """Generate all transactions (normal + fraud) with realistic patterns"""
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

        if self.use_realistic_generation and self.account_profiles:
            print(f"Generating {normal_count} realistic normal transactions...")
            normal_transactions = self._generate_realistic_normal_transactions(normal_count, tx_id)
        else:
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

    def _generate_realistic_normal_transactions(self, target_count: int, starting_tx_id: int) -> List[Dict[str, Any]]:
        """
        Generate realistic normal transactions based on account profiles

        NEW: Generates transactions according to account frequency profiles and supports
        recurring transactions, transaction chains, and batch payments
        """
        temporal_config = self.config.get('temporal_patterns', {})
        enable_recurring = temporal_config.get('recurring_transactions', {}).get('enable', False)
        enable_chains = temporal_config.get('transaction_chains', {}).get('enable', False)
        enable_batches = temporal_config.get('batch_payments', {}).get('enable', False)

        transactions = []
        tx_id = starting_tx_id

        # Phase 1: Generate recurring transactions (if enabled)
        recurring_transactions = []
        if enable_recurring:
            print("  📅 Generating recurring transaction patterns...")
            recurring_transactions = self._generate_recurring_transactions(tx_id)
            tx_id += len(recurring_transactions)
            transactions.extend(recurring_transactions)
            print(f"    Generated {len(recurring_transactions)} recurring transactions")

        # Phase 2: Generate account-based transactions
        remaining_count = target_count - len(transactions)
        print(f"  🎲 Generating {remaining_count} profile-based transactions...")

        # Allocate transactions to accounts based on their frequency profiles
        account_tx_allocations = self._allocate_transactions_to_accounts(remaining_count)

        for account_id, tx_count in account_tx_allocations.items():
            sender = next(acc for acc in self.accounts if acc['account_id'] == account_id)
            profile = self.account_profiles[account_id]

            for _ in range(tx_count):
                # Check for batch payment generation
                if enable_batches and self._should_generate_batch(profile):
                    batch_txs = self._generate_batch_payments(sender, profile, tx_id)
                    transactions.extend(batch_txs)
                    tx_id += len(batch_txs)
                else:
                    # Generate single transaction
                    tx = self._generate_normal_transaction(tx_id, sender=sender)
                    transactions.append(tx)

                    # Check for transaction chain
                    if enable_chains and self._should_generate_chain():
                        chain_txs = self._generate_transaction_chain(tx, tx_id + 1)
                        transactions.extend(chain_txs)
                        tx_id += len(chain_txs)

                    tx_id += 1

        # Trim to exact target count if we overgenerated
        if len(transactions) > target_count:
            transactions = transactions[:target_count]

        print(f"  ✅ Generated {len(transactions)} total realistic transactions")

        return transactions

    def _allocate_transactions_to_accounts(self, total_count: int) -> Dict[str, int]:
        """Allocate transactions to accounts based on their frequency profiles"""
        time_period_days = self.config['transaction_generation']['time_period_days']

        # Calculate expected transactions per account
        expected_txs = {}
        total_expected = 0

        for account_id, profile in self.account_profiles.items():
            expected = profile.get_transaction_count(time_period_days)
            expected_txs[account_id] = expected
            total_expected += expected

        # Normalize to match target count
        allocations = {}
        if total_expected > 0:
            for account_id, expected in expected_txs.items():
                allocation = int((expected / total_expected) * total_count)
                allocations[account_id] = allocation
        else:
            # Fallback: distribute evenly
            accounts_per_tx = max(1, len(self.account_profiles) // total_count)
            for i, account_id in enumerate(self.account_profiles.keys()):
                allocations[account_id] = 1 if i < total_count else 0

        return allocations

    def _generate_recurring_transactions(self, starting_tx_id: int) -> List[Dict[str, Any]]:
        """Generate recurring transaction patterns (monthly, weekly, bi-weekly)"""
        temporal_config = self.config.get('temporal_patterns', {}).get('recurring_transactions', {})
        monthly_prob = temporal_config.get('monthly_probability', 0.3)
        weekly_prob = temporal_config.get('weekly_probability', 0.15)
        biweekly_prob = temporal_config.get('biweekly_probability', 0.1)

        time_period_days = self.config['transaction_generation']['time_period_days']
        base_date = datetime.strptime(self.config['transaction_generation']['base_date'], '%Y-%m-%d')

        transactions = []
        tx_id = starting_tx_id

        # Go through relationships and create recurring patterns
        for sender_id, relationships in self.relationship_network.relationships.items():
            sender = next(acc for acc in self.accounts if acc['account_id'] == sender_id)
            sender_profile = self.account_profiles[sender_id]

            for rel in relationships:
                receiver_id = rel['receiver_id']
                rel_type = rel['type']
                receiver = next(acc for acc in self.accounts if acc['account_id'] == receiver_id)

                # Determine if this relationship generates recurring transactions
                if rel_type in [RelationshipType.RECURRING_PAYMENT, RelationshipType.BILL_PAYMENT, RelationshipType.SALARY_DEPOSIT]:
                    # Decide frequency
                    rand = random.random()
                    if rand < monthly_prob:
                        frequency = 'monthly'
                        interval_days = 30
                    elif rand < monthly_prob + weekly_prob:
                        frequency = 'weekly'
                        interval_days = 7
                    elif rand < monthly_prob + weekly_prob + biweekly_prob:
                        frequency = 'biweekly'
                        interval_days = 14
                    else:
                        continue  # No recurring pattern for this relationship

                    # Generate recurring transactions
                    num_occurrences = time_period_days // interval_days
                    for occurrence in range(num_occurrences):
                        day_offset = occurrence * interval_days + random.randint(0, 2)  # Small variance
                        if day_offset >= time_period_days:
                            break

                        # Generate timestamp on specific day
                        date = base_date + timedelta(days=day_offset)
                        hour = random.choice(sender_profile.preferred_hours)
                        timestamp = datetime(date.year, date.month, date.day, hour, random.randint(0, 59))

                        tx = self._generate_normal_transaction(
                            tx_id,
                            sender=sender,
                            receiver=receiver,
                            relationship_type=rel_type,
                            timestamp=timestamp
                        )
                        transactions.append(tx)
                        tx_id += 1

        return transactions

    def _should_generate_batch(self, profile: AccountBehaviorProfile) -> bool:
        """Determine if account should generate batch payments"""
        temporal_config = self.config.get('temporal_patterns', {}).get('batch_payments', {})
        batch_prob = temporal_config.get('batch_probability', 0.08)

        # Only high-frequency accounts generate batches
        if profile.frequency_tier in [FrequencyTier.HIGH, FrequencyTier.VERY_HIGH]:
            return random.random() < batch_prob
        return False

    def _generate_batch_payments(self, sender: Dict, profile: AccountBehaviorProfile, starting_tx_id: int) -> List[Dict[str, Any]]:
        """Generate a batch of payments from the same sender"""
        temporal_config = self.config.get('temporal_patterns', {}).get('batch_payments', {})
        batch_size_range = temporal_config.get('batch_size_range', [3, 10])
        time_window_minutes = temporal_config.get('batch_time_window_minutes', [5, 60])

        batch_size = random.randint(*batch_size_range)
        window = random.randint(*time_window_minutes)

        # Generate base timestamp
        base_timestamp = self._generate_realistic_timestamp(profile)

        batch_txs = []
        for i in range(batch_size):
            # Generate transaction within time window
            minute_offset = random.randint(0, window)
            timestamp = base_timestamp + timedelta(minutes=minute_offset)

            tx = self._generate_normal_transaction(
                starting_tx_id + i,
                sender=sender,
                timestamp=timestamp
            )
            batch_txs.append(tx)

        return batch_txs

    def _should_generate_chain(self) -> bool:
        """Determine if transaction should trigger a chain"""
        temporal_config = self.config.get('temporal_patterns', {}).get('transaction_chains', {})
        chain_prob = temporal_config.get('chain_probability', 0.15)
        return random.random() < chain_prob

    def _generate_transaction_chain(self, initial_tx: Dict, starting_tx_id: int) -> List[Dict[str, Any]]:
        """Generate a chain of related transactions"""
        temporal_config = self.config.get('temporal_patterns', {}).get('transaction_chains', {})
        chain_length_range = temporal_config.get('chain_length_range', [2, 4])
        time_window_hours = temporal_config.get('chain_time_window_hours', [1, 48])

        chain_length = random.randint(*chain_length_range) - 1  # -1 because initial_tx is first
        time_window = random.randint(*time_window_hours)

        chain_txs = []
        initial_timestamp = datetime.strptime(initial_tx['timestamp'], '%Y-%m-%d %H:%M:%S')

        # Chain starts from receiver of previous transaction
        current_sender_id = initial_tx['receiver_account']

        for i in range(chain_length):
            current_sender = next(acc for acc in self.accounts if acc['account_id'] == current_sender_id)

            # Generate timestamp within window
            hour_offset = random.randint(1, time_window)
            timestamp = initial_timestamp + timedelta(hours=hour_offset)

            tx = self._generate_normal_transaction(
                starting_tx_id + i,
                sender=current_sender,
                timestamp=timestamp
            )
            chain_txs.append(tx)

            # Next sender is current receiver
            current_sender_id = tx['receiver_account']

        return chain_txs
    
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
    
    def _generate_normal_transaction(self, tx_id: int, sender: Optional[Dict] = None,
                                     receiver: Optional[Dict] = None,
                                     relationship_type: Optional[RelationshipType] = None,
                                     timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate normal transaction with realistic patterns

        NEW: Supports relationship-based generation, account behavior profiles,
        realistic amount distributions, and temporal patterns
        """
        tx_config = self.config['transaction_generation']['normal_transactions']

        # Use realistic generation if enabled
        if self.use_realistic_generation and self.account_profiles:
            return self._generate_realistic_normal_transaction(tx_id, sender, receiver, relationship_type, timestamp)

        # FALLBACK: Original simple generation for backward compatibility
        if sender is None:
            sender = random.choice(self.accounts)
        if receiver is None:
            receiver = random.choice(self.accounts)
            while sender['account_id'] == receiver['account_id']:
                receiver = random.choice(self.accounts)

        # Generate amount
        amount = random.uniform(*tx_config['amount_range'])

        # Generate timestamp with business hour weighting
        if timestamp is None:
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
        risk_score = self._calculate_risk_score(amount, timestamp.hour, None, sender, receiver)

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
            'is_fraud': False,
            'fraud_type': '',
            'risk_score': risk_score
        }

    def _generate_realistic_normal_transaction(self, tx_id: int,
                                               sender: Optional[Dict] = None,
                                               receiver: Optional[Dict] = None,
                                               relationship_type: Optional[RelationshipType] = None,
                                               timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate realistic normal transaction using account profiles and relationships"""
        tx_config = self.config['transaction_generation']['normal_transactions']

        # Select sender account if not provided
        if sender is None:
            sender = random.choice(self.accounts)

        sender_profile = self.account_profiles[sender['account_id']]

        # Select receiver based on relationship network AND institutional relationships
        if receiver is None:
            # First, check if we should use institutional relationships for institution selection
            use_institutional_matrix = (
                self.institutional_relationships and
                self.institutional_relationships.is_loaded and
                self.config.get('institutional_relationships', {}).get('use_matrix', True)
            )

            if use_institutional_matrix:
                # Use ODFI-RDFI matrix to select receiver institution
                receiver = self._select_receiver_with_institutional_matrix(sender, relationship_type, tx_config)
            else:
                # Original logic: select receiver from account relationship network
                relationship_prob = tx_config.get('relationship_based_probability', 0.8)
                receiver_id, relationship_type = self.relationship_network.select_receiver(
                    sender['account_id'],
                    use_relationship_prob=relationship_prob
                )

                if receiver_id:
                    receiver = next(acc for acc in self.accounts if acc['account_id'] == receiver_id)
                else:
                    # No relationship - select random receiver
                    receiver = random.choice(self.accounts)
                    while sender['account_id'] == receiver['account_id']:
                        receiver = random.choice(self.accounts)

        # Generate realistic amount based on account profile and relationship type
        amount = self._generate_realistic_amount(sender_profile, relationship_type)

        # Generate realistic timestamp based on account profile
        if timestamp is None:
            timestamp = self._generate_realistic_timestamp(sender_profile)

        # Select payment type based on amount and relationship
        payment_type = self._select_payment_type(amount, relationship_type, tx_config)

        # Calculate risk score
        risk_score = self._calculate_risk_score(amount, timestamp.hour, None, sender, receiver)

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
            'is_fraud': False,
            'fraud_type': '',
            'risk_score': risk_score,
            'relationship_type': relationship_type.value if relationship_type else 'one_time'
        }

    def _select_receiver_with_institutional_matrix(self, sender: Dict,
                                                            relationship_type: Optional[RelationshipType],
                                                            tx_config: Dict) -> Dict:
        """
        Select receiver account using institutional relationship matrix

        Uses real-world ODFI->RDFI patterns to select receiver institution,
        then selects account from that institution
        """
        # Get sender's institution
        sender_inst = next(inst for inst in self.institutions if inst['institution_id'] == sender['institution_id'])
        sender_routing = sender_inst.get('routing_number', sender_inst['institution_id'])

        # Get receiver institution based on ODFI->RDFI patterns
        receiver_routing, inst_relationship = self.institutional_relationships.select_rdfi_for_odfi(
            sender_routing,
            fallback_institutions=[inst['routing_number'] for inst in self.institutions]
        )

        # Find the institution object
        receiver_inst = None
        for inst in self.institutions:
            if inst.get('routing_number') == receiver_routing:
                receiver_inst = inst
                break

        if not receiver_inst:
            # Fallback to random institution
            receiver_inst = random.choice([inst for inst in self.institutions
                                          if inst['institution_id'] != sender['institution_id']])

        # Get all accounts at receiver institution
        receiver_inst_accounts = [
            acc for acc in self.accounts
            if acc['institution_id'] == receiver_inst['institution_id']
            and acc['account_id'] != sender['account_id']
        ]

        if not receiver_inst_accounts:
            # Fallback: select any account except sender
            receiver_inst_accounts = [acc for acc in self.accounts if acc['account_id'] != sender['account_id']]

        # Within the receiver institution, prefer accounts with existing relationships
        relationship_prob = tx_config.get('relationship_based_probability', 0.8)
        if random.random() < relationship_prob and self.relationship_network:
            # Try to find receiver from relationship network within this institution
            receiver_id, rel_type = self.relationship_network.select_receiver(
                sender['account_id'],
                use_relationship_prob=1.0  # Always use relationship if available
            )

            if receiver_id:
                # Check if this receiver is in our target institution
                potential_receiver = next((acc for acc in receiver_inst_accounts
                                          if acc['account_id'] == receiver_id), None)
                if potential_receiver:
                    if relationship_type is None:
                        relationship_type = rel_type
                    return potential_receiver

        # No relationship match, select random account from receiver institution
        return random.choice(receiver_inst_accounts)

    def _generate_realistic_amount(self, profile: AccountBehaviorProfile,
                                   relationship_type: Optional[RelationshipType] = None) -> float:
        """Generate realistic transaction amount based on profile and relationship"""
        # Base amount from profile (uses log-normal distribution)
        base_amount = profile.generate_amount()

        # Adjust based on relationship type
        if relationship_type == RelationshipType.BILL_PAYMENT:
            # Bill payments tend to be consistent amounts
            base_amount = round(base_amount / 50) * 50  # Round to nearest $50
        elif relationship_type == RelationshipType.SALARY_DEPOSIT:
            # Salaries are larger, consistent amounts
            base_amount = max(base_amount * 3, 1500)
            base_amount = round(base_amount / 100) * 100  # Round to nearest $100
        elif relationship_type == RelationshipType.PEER_TRANSFER:
            # Peer transfers tend to be smaller, varied amounts
            base_amount = base_amount * 0.6
        elif relationship_type == RelationshipType.BUSINESS_PAYMENT:
            # Business payments can be larger
            base_amount = base_amount * 1.5

        return max(10.0, base_amount)

    def _generate_realistic_timestamp(self, profile: AccountBehaviorProfile) -> datetime:
        """Generate realistic timestamp based on account behavior profile"""
        base_config = self.config['transaction_generation']
        tx_config = base_config['normal_transactions']
        base_date = datetime.strptime(base_config['base_date'], '%Y-%m-%d')
        days_range = base_config['time_period_days']

        # Select day with day-of-week weighting
        day_weights = tx_config.get('day_of_week_weights', [1.0] * 7)

        # Generate random day
        random_day = random.randint(0, days_range - 1)
        date = base_date + timedelta(days=random_day)
        day_of_week = date.weekday()

        # Check if this day aligns with profile preferences
        # Reroll if needed (with probability)
        if not profile.is_active_day(day_of_week) and random.random() < 0.7:
            # 70% chance to reroll if day doesn't match preference
            for _ in range(3):  # Try up to 3 times
                random_day = random.randint(0, days_range - 1)
                date = base_date + timedelta(days=random_day)
                day_of_week = date.weekday()
                if profile.is_active_day(day_of_week):
                    break

        # Select hour based on account profile preferences
        preferred_hours = profile.preferred_hours
        if random.random() < 0.8:  # 80% chance to use preferred hours
            hour = random.choice(preferred_hours)
        else:
            # Occasional activity outside preferred hours
            hour = random.randint(6, 22)

        # Generate minute with some clustering
        if tx_config.get('intra_day_clustering', False):
            # Cluster around common times (on the hour, half hour, quarter hour)
            cluster_minute = random.choice([0, 15, 30, 45])
            minute = cluster_minute + random.randint(-5, 5)
            minute = max(0, min(59, minute))
        else:
            minute = random.randint(0, 59)

        return datetime(date.year, date.month, date.day, hour, minute)

    def _select_payment_type(self, amount: float, relationship_type: Optional[RelationshipType],
                            tx_config: Dict) -> str:
        """Select payment type based on amount and relationship"""
        payment_dist = tx_config['payment_type_distribution']

        # Modify distribution based on amount and relationship
        if amount > 10000:
            # Large amounts favor WIRE
            return 'WIRE' if random.random() < 0.7 else 'ACH'
        elif relationship_type == RelationshipType.BILL_PAYMENT:
            # Bill payments favor ACH
            return 'ACH' if random.random() < 0.9 else random.choice(['WIRE', 'RTP'])
        elif relationship_type == RelationshipType.PEER_TRANSFER:
            # Peer transfers favor RTP (real-time)
            return 'RTP' if random.random() < 0.6 else 'ACH'
        else:
            # Use configured distribution
            return np.random.choice(
                list(payment_dist.keys()),
                p=list(payment_dist.values())
            )
    
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

    def _map_transactions_to_schema(self):
        """Map all transactions to full payment system schema"""
        if not self.schema_mapper:
            print("  ⚠️  Schema mapper not initialized, skipping schema mapping")
            return

        # Create account lookup dictionary for faster access
        accounts_dict = {acc['account_id']: acc for acc in self.accounts}

        # Map each transaction to full schema
        mapped_transactions = []
        for tx in self.transactions:
            try:
                schema_tx = self.schema_mapper.map_to_schema(tx, accounts_dict)
                mapped_transactions.append(schema_tx)
            except Exception as e:
                print(f"  ⚠️  Error mapping transaction {tx.get('transaction_id', 'unknown')}: {e}")
                # Keep original transaction if mapping fails
                mapped_transactions.append(tx)

        # Replace transactions with schema-compliant versions
        self.transactions = mapped_transactions

        print(f"  ✅ Mapped {len(mapped_transactions)} transactions to full schema")

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

        # Use dollar_value if available (schema-compliant), otherwise fallback to amount
        amount_field = 'dollar_value' if 'dollar_value' in transactions_df.columns else 'amount'
        payment_type_field = 'payment_rail' if 'payment_rail' in transactions_df.columns else 'payment_type'

        summary = {
            'total_transactions': len(self.transactions),
            'fraud_transactions': len(fraud_df),
            'fraud_rate': len(fraud_df) / len(self.transactions) * 100 if len(self.transactions) > 0 else 0,
            'total_fraud_amount': float(fraud_df[amount_field].sum()) if len(fraud_df) > 0 else 0,
            'avg_fraud_amount': float(fraud_df[amount_field].mean()) if len(fraud_df) > 0 else 0,
            'fraud_by_type': fraud_df['fraud_type'].value_counts().to_dict() if len(fraud_df) > 0 else {},
            'fraud_by_payment_type': fraud_df[payment_type_field].value_counts().to_dict() if len(fraud_df) > 0 else {},
            'avg_risk_score': float(transactions_df['risk_score'].mean()) if 'risk_score' in transactions_df.columns else 0,
            'high_risk_transactions': len(transactions_df[transactions_df['risk_score'] >= 70]) if 'risk_score' in transactions_df.columns else 0
        }

        with open(f'{output_dir}/fraud_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)