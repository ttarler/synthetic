import pandas as pd
import boto3
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.traversal import T
from typing import Dict, List
import json

class EnhancedNeptuneLoader:
    def __init__(self, neptune_endpoint: str):
        self.endpoint = neptune_endpoint
        self.connection = DriverRemoteConnection(f'wss://{neptune_endpoint}:8182/gremlin', 'g')
        self.g = traversal().withRemote(self.connection)
        
    def load_enhanced_fraud_network(self, data_dir: str = "enhanced_output"):
        """Load enhanced fraud network data into Neptune"""
        print("🌊 Loading enhanced fraud network into Neptune...")
        
        # Load datasets
        institutions_df = pd.read_csv(f'{data_dir}/institutions.csv')
        accounts_df = pd.read_csv(f'{data_dir}/accounts.csv')
        transactions_df = pd.read_csv(f'{data_dir}/transactions.csv')
        
        # Clear existing data
        print("🧹 Clearing existing graph data...")
        self._clear_graph()
        
        # Load institutions
        print("🏦 Loading institutions...")
        self._load_institutions(institutions_df)
        
        # Load accounts
        print("👤 Loading accounts...")
        self._load_accounts(accounts_df)
        
        # Create institution-account relationships
        print("🔗 Creating institution-account relationships...")
        self._create_institution_relationships(accounts_df)
        
        # Load transactions in batches
        print("💸 Loading transactions...")
        self._load_transactions_batch(transactions_df)
        
        # Create fraud indicators
        print("🚨 Creating fraud indicators...")
        self._create_enhanced_fraud_indicators(transactions_df)
        
        # Create fraud rings
        print("🕸️ Creating fraud ring structures...")
        self._create_fraud_ring_structures(transactions_df)
        
        print("✅ Enhanced fraud network loaded successfully!")
        
    def _clear_graph(self):
        """Clear all existing graph data"""
        self.g.V().drop().iterate()
        self.g.E().drop().iterate()
        
    def _load_institutions(self, df: pd.DataFrame):
        """Load institutions as vertices"""
        for _, row in df.iterrows():
            self.g.addV('Institution')\
                .property(T.id, row['institution_id'])\
                .property('name', row['name'])\
                .property('type', row['type'])\
                .property('channels', row['channels']).next()
                
    def _load_accounts(self, df: pd.DataFrame):
        """Load accounts as vertices"""
        for _, row in df.iterrows():
            vertex = self.g.addV('Account')\
                .property(T.id, row['account_id'])\
                .property('account_type', row['account_type'])\
                .property('balance', row['balance'])\
                .property('is_shell', row['is_shell'])\
                .property('is_synthetic', row['is_synthetic'])
            
            # Add risk indicators
            if row['is_shell']:
                vertex = vertex.property('risk_category', 'shell_account')
            elif row['is_synthetic']:
                vertex = vertex.property('risk_category', 'synthetic_identity')
            else:
                vertex = vertex.property('risk_category', 'normal')
                
            vertex.next()
            
    def _create_institution_relationships(self, accounts_df: pd.DataFrame):
        """Create BELONGS_TO relationships between accounts and institutions"""
        for _, row in accounts_df.iterrows():
            self.g.V(row['account_id'])\
                .addE('BELONGS_TO')\
                .to(__.V(row['institution_id']))\
                .property('relationship_type', 'account_holder').next()
                
    def _load_transactions_batch(self, df: pd.DataFrame, batch_size: int = 1000):
        """Load transactions in batches for better performance"""
        total_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_num = i // batch_size + 1
            print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} transactions)")
            
            for _, row in batch.iterrows():
                edge = self.g.V(row['sender_account'])\
                    .addE('PAYMENT')\
                    .to(__.V(row['receiver_account']))\
                    .property('transaction_id', row['transaction_id'])\
                    .property('amount', row['amount'])\
                    .property('payment_type', row['payment_type'])\
                    .property('timestamp', row['timestamp'])\
                    .property('hour', row['hour'])\
                    .property('is_fraud', row['is_fraud'])\
                    .property('risk_score', row['risk_score'])
                
                if pd.notna(row['fraud_type']) and row['fraud_type']:
                    edge = edge.property('fraud_type', row['fraud_type'])
                    
                edge.next()
                
    def _create_enhanced_fraud_indicators(self, df: pd.DataFrame):
        """Create enhanced fraud indicator vertices"""
        fraud_df = df[df['is_fraud'] == True]
        
        # Group by fraud type
        fraud_groups = fraud_df.groupby('fraud_type')
        
        for fraud_type, group in fraud_groups:
            # Create fraud pattern vertex
            pattern_id = f"PATTERN_{fraud_type.upper()}"
            self.g.addV('FraudPattern')\
                .property(T.id, pattern_id)\
                .property('pattern_type', fraud_type)\
                .property('transaction_count', len(group))\
                .property('total_amount', group['amount'].sum())\
                .property('avg_amount', group['amount'].mean())\
                .property('avg_risk_score', group['risk_score'].mean()).next()
            
            # Link transactions to pattern
            for _, row in group.iterrows():
                # Create fraud indicator for each transaction
                indicator_id = f"FRAUD_{row['transaction_id']}"
                self.g.addV('FraudIndicator')\
                    .property(T.id, indicator_id)\
                    .property('transaction_id', row['transaction_id'])\
                    .property('fraud_type', row['fraud_type'])\
                    .property('amount', row['amount'])\
                    .property('risk_score', row['risk_score'])\
                    .property('timestamp', row['timestamp']).next()
                
                # Link to pattern
                self.g.V(indicator_id)\
                    .addE('BELONGS_TO_PATTERN')\
                    .to(__.V(pattern_id)).next()
                    
    def _create_fraud_ring_structures(self, df: pd.DataFrame):
        """Create fraud ring structures for money laundering detection"""
        ml_transactions = df[df['fraud_type'] == 'money_laundering_ring']
        
        if len(ml_transactions) == 0:
            return
            
        # Group by potential rings (accounts that transact with each other)
        ring_accounts = {}
        
        for _, row in ml_transactions.iterrows():
            sender = row['sender_account']
            receiver = row['receiver_account']
            
            # Find or create ring
            ring_id = None
            for rid, accounts in ring_accounts.items():
                if sender in accounts or receiver in accounts:
                    ring_id = rid
                    break
            
            if ring_id is None:
                ring_id = f"RING_{len(ring_accounts):03d}"
                ring_accounts[ring_id] = set()
            
            ring_accounts[ring_id].add(sender)
            ring_accounts[ring_id].add(receiver)
        
        # Create ring vertices and relationships
        for ring_id, accounts in ring_accounts.items():
            if len(accounts) >= 3:  # Only create rings with 3+ accounts
                # Create ring vertex
                ring_transactions = ml_transactions[
                    (ml_transactions['sender_account'].isin(accounts)) &
                    (ml_transactions['receiver_account'].isin(accounts))
                ]
                
                self.g.addV('FraudRing')\
                    .property(T.id, ring_id)\
                    .property('ring_type', 'money_laundering')\
                    .property('account_count', len(accounts))\
                    .property('transaction_count', len(ring_transactions))\
                    .property('total_amount', ring_transactions['amount'].sum()).next()
                
                # Link accounts to ring
                for account_id in accounts:
                    self.g.V(account_id)\
                        .addE('MEMBER_OF_RING')\
                        .to(__.V(ring_id))\
                        .property('role', 'participant').next()
                        
    def detect_enhanced_fraud_patterns(self) -> Dict[str, List]:
        """Detect various enhanced fraud patterns using graph queries"""
        patterns = {}
        
        # Money laundering rings
        patterns['money_laundering_rings'] = self._find_money_laundering_rings()
        
        # Shell account networks
        patterns['shell_networks'] = self._find_shell_account_networks()
        
        # Synthetic identity patterns
        patterns['synthetic_identity_patterns'] = self._find_synthetic_identity_patterns()
        
        # High-risk transaction chains
        patterns['high_risk_chains'] = self._find_high_risk_chains()
        
        # Unusual amount patterns
        patterns['unusual_amounts'] = self._find_unusual_amount_patterns()
        
        return patterns
        
    def _find_money_laundering_rings(self) -> List[Dict]:
        """Find money laundering ring patterns"""
        results = self.g.V().hasLabel('FraudRing')\
            .has('ring_type', 'money_laundering')\
            .project('ring_id', 'accounts', 'transaction_count', 'total_amount')\
            .by(T.id)\
            .by(__.in_('MEMBER_OF_RING').id().fold())\
            .by('transaction_count')\
            .by('total_amount').toList()
        return results
        
    def _find_shell_account_networks(self) -> List[Dict]:
        """Find shell account transaction networks"""
        results = self.g.V().has('Account', 'is_shell', True)\
            .as_('shell')\
            .bothE('PAYMENT').as_('transaction')\
            .otherV().as_('connected')\
            .select('shell', 'transaction', 'connected')\
            .by(T.id)\
            .by(__.valueMap())\
            .by(T.id).limit(100).toList()
        return results
        
    def _find_synthetic_identity_patterns(self) -> List[Dict]:
        """Find synthetic identity fraud patterns"""
        results = self.g.V().has('Account', 'is_synthetic', True)\
            .as_('synthetic')\
            .outE('PAYMENT').has('is_fraud', True)\
            .as_('fraud_tx')\
            .inV().as_('receiver')\
            .select('synthetic', 'fraud_tx', 'receiver')\
            .by(T.id)\
            .by(__.valueMap())\
            .by(T.id).limit(50).toList()
        return results
        
    def _find_high_risk_chains(self) -> List[Dict]:
        """Find chains of high-risk transactions"""
        results = self.g.V().as_('start')\
            .outE('PAYMENT').has('risk_score', __.gte(70)).as_('tx1')\
            .inV().outE('PAYMENT').has('risk_score', __.gte(70)).as_('tx2')\
            .inV().as_('end')\
            .path().by(T.id).by('risk_score').limit(50).toList()
        return results
        
    def _find_unusual_amount_patterns(self) -> List[Dict]:
        """Find unusual amount transaction patterns"""
        results = self.g.E().hasLabel('PAYMENT')\
            .has('fraud_type', 'unusual_amount')\
            .project('transaction_id', 'sender', 'receiver', 'amount', 'risk_score')\
            .by('transaction_id')\
            .by(__.outV().id())\
            .by(__.inV().id())\
            .by('amount')\
            .by('risk_score').limit(100).toList()
        return results
        
    def get_enhanced_risk_metrics(self) -> Dict:
        """Calculate enhanced risk metrics for the network"""
        metrics = {}
        
        # Overall fraud statistics
        total_transactions = self.g.E().hasLabel('PAYMENT').count().next()
        fraud_transactions = self.g.E().hasLabel('PAYMENT').has('is_fraud', True).count().next()
        
        metrics['total_transactions'] = total_transactions
        metrics['fraud_transactions'] = fraud_transactions
        metrics['fraud_rate'] = fraud_transactions / total_transactions * 100 if total_transactions > 0 else 0
        
        # Risk score distribution
        high_risk = self.g.E().hasLabel('PAYMENT').has('risk_score', __.gte(70)).count().next()
        medium_risk = self.g.E().hasLabel('PAYMENT').has('risk_score', __.gte(40)).has('risk_score', __.lt(70)).count().next()
        
        metrics['high_risk_transactions'] = high_risk
        metrics['medium_risk_transactions'] = medium_risk
        metrics['high_risk_percentage'] = high_risk / total_transactions * 100 if total_transactions > 0 else 0
        
        # Fraud pattern counts
        fraud_patterns = self.g.V().hasLabel('FraudPattern').valueMap().toList()
        metrics['fraud_patterns'] = fraud_patterns
        
        # Shell and synthetic account activity
        shell_transactions = self.g.V().has('Account', 'is_shell', True).bothE('PAYMENT').count().next()
        synthetic_transactions = self.g.V().has('Account', 'is_synthetic', True).bothE('PAYMENT').count().next()
        
        metrics['shell_account_transactions'] = shell_transactions
        metrics['synthetic_account_transactions'] = synthetic_transactions
        
        return metrics
        
    def close(self):
        """Close Neptune connection"""
        self.connection.close()