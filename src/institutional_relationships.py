"""
Institutional Relationships Manager

Manages real-world transaction volume patterns between financial institutions.
Uses ODFI-RDFI matrix data to generate realistic institution selections based
on actual observed money flows.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import os


class InstitutionalRelationships:
    """Manages institutional relationship matrix for realistic transaction generation"""

    def __init__(self, matrix_path: Optional[str] = None):
        """
        Initialize institutional relationships

        Args:
            matrix_path: Path to CSV file containing ODFI-RDFI matrix
                        Expected columns: odfi, rdfi, transaction_count, total_dollar_value,
                        avg_dollar_value, ach_percentage, wire_percentage, rtp_percentage,
                        recurring_percentage, relationship_strength
        """
        self.matrix_path = matrix_path
        self.matrix_df: Optional[pd.DataFrame] = None
        self.odfi_to_rdfi: Dict[str, List[Dict]] = defaultdict(list)
        self.rdfi_to_odfi: Dict[str, List[Dict]] = defaultdict(list)
        self.all_institutions: set = set()
        self.is_loaded = False

        if matrix_path and os.path.exists(matrix_path):
            self.load_matrix(matrix_path)

    def load_matrix(self, matrix_path: str):
        """Load ODFI-RDFI relationship matrix from CSV"""
        try:
            self.matrix_df = pd.read_csv(matrix_path)
            print(f"📊 Loading institutional relationship matrix from {matrix_path}")

            # Validate required columns
            required_cols = ['odfi', 'rdfi', 'transaction_count', 'relationship_strength']
            missing_cols = [col for col in required_cols if col not in self.matrix_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in matrix: {missing_cols}")

            # Build lookup dictionaries
            self._build_lookups()

            print(f"   ✓ Loaded {len(self.matrix_df)} institutional relationships")
            print(f"   ✓ {len(self.all_institutions)} unique institutions")
            print(f"   ✓ Avg {len(self.matrix_df) / len(self.all_institutions):.1f} relationships per institution")

            self.is_loaded = True

        except FileNotFoundError:
            print(f"   ⚠️  Matrix file not found: {matrix_path}")
            print(f"   → Generate matrix using queries/generate_odfi_rdfi_matrix.sql")
            self.is_loaded = False
        except Exception as e:
            print(f"   ⚠️  Error loading matrix: {e}")
            self.is_loaded = False

    def _build_lookups(self):
        """Build fast lookup dictionaries from matrix"""
        for _, row in self.matrix_df.iterrows():
            odfi = str(row['odfi'])
            rdfi = str(row['rdfi'])

            # Track all institutions
            self.all_institutions.add(odfi)
            self.all_institutions.add(rdfi)

            # Build relationship data
            relationship = {
                'rdfi': rdfi,
                'odfi': odfi,
                'transaction_count': row['transaction_count'],
                'total_dollar_value': row.get('total_dollar_value', 0),
                'avg_dollar_value': row.get('avg_dollar_value', 0),
                'ach_percentage': row.get('ach_percentage', 60.0),
                'wire_percentage': row.get('wire_percentage', 15.0),
                'rtp_percentage': row.get('rtp_percentage', 25.0),
                'recurring_percentage': row.get('recurring_percentage', 30.0),
                'relationship_strength': row.get('relationship_strength', 0.5)
            }

            # ODFI -> RDFI mapping
            self.odfi_to_rdfi[odfi].append(relationship)

            # RDFI -> ODFI mapping (reverse)
            reverse_relationship = relationship.copy()
            reverse_relationship['odfi'] = odfi
            self.rdfi_to_odfi[rdfi].append(reverse_relationship)

    def select_rdfi_for_odfi(self, odfi: str, fallback_institutions: List[str] = None) -> Tuple[str, Dict]:
        """
        Select RDFI (receiving institution) for a given ODFI (sending institution)
        based on observed transaction volumes

        Args:
            odfi: The originating institution routing number
            fallback_institutions: List of institutions to use if ODFI not in matrix

        Returns:
            Tuple of (rdfi_routing, relationship_metadata)
        """
        if not self.is_loaded or odfi not in self.odfi_to_rdfi:
            # No data available - random selection from fallback
            if fallback_institutions:
                rdfi = np.random.choice(fallback_institutions)
                return rdfi, self._default_relationship_metadata()
            return None, {}

        # Get all RDFI relationships for this ODFI
        relationships = self.odfi_to_rdfi[odfi]

        # Weight by relationship strength (or transaction count)
        weights = [rel['relationship_strength'] for rel in relationships]

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            probabilities = [1.0 / len(weights)] * len(weights)
        else:
            probabilities = [w / total_weight for w in weights]

        # Sample based on probabilities
        selected_idx = np.random.choice(len(relationships), p=probabilities)
        selected_relationship = relationships[selected_idx]

        return selected_relationship['rdfi'], selected_relationship

    def select_odfi_for_rdfi(self, rdfi: str, fallback_institutions: List[str] = None) -> Tuple[str, Dict]:
        """
        Select ODFI (sending institution) for a given RDFI (receiving institution)
        Useful for reverse lookups

        Args:
            rdfi: The receiving institution routing number
            fallback_institutions: List of institutions to use if RDFI not in matrix

        Returns:
            Tuple of (odfi_routing, relationship_metadata)
        """
        if not self.is_loaded or rdfi not in self.rdfi_to_odfi:
            if fallback_institutions:
                odfi = np.random.choice(fallback_institutions)
                return odfi, self._default_relationship_metadata()
            return None, {}

        relationships = self.rdfi_to_odfi[rdfi]
        weights = [rel['relationship_strength'] for rel in relationships]

        total_weight = sum(weights)
        if total_weight == 0:
            probabilities = [1.0 / len(weights)] * len(weights)
        else:
            probabilities = [w / total_weight for w in weights]

        selected_idx = np.random.choice(len(relationships), p=probabilities)
        selected_relationship = relationships[selected_idx]

        return selected_relationship['odfi'], selected_relationship

    def get_payment_rail_distribution(self, relationship_metadata: Dict) -> Dict[str, float]:
        """
        Get payment rail distribution for a specific institutional relationship

        Args:
            relationship_metadata: Metadata from select_rdfi_for_odfi

        Returns:
            Dictionary with payment rail probabilities
        """
        if not relationship_metadata:
            return {'ACH': 0.60, 'WIRE': 0.15, 'RTP': 0.25}

        # Convert percentages to probabilities
        ach_prob = relationship_metadata.get('ach_percentage', 60.0) / 100.0
        wire_prob = relationship_metadata.get('wire_percentage', 15.0) / 100.0
        rtp_prob = relationship_metadata.get('rtp_percentage', 25.0) / 100.0

        # Normalize to ensure they sum to 1.0
        total = ach_prob + wire_prob + rtp_prob
        if total > 0:
            return {
                'ACH': ach_prob / total,
                'WIRE': wire_prob / total,
                'RTP': rtp_prob / total
            }
        else:
            return {'ACH': 0.60, 'WIRE': 0.15, 'RTP': 0.25}

    def should_be_recurring(self, relationship_metadata: Dict) -> bool:
        """
        Determine if transaction should be recurring based on institutional relationship

        Args:
            relationship_metadata: Metadata from select_rdfi_for_odfi

        Returns:
            True if transaction should be recurring
        """
        if not relationship_metadata:
            return False

        recurring_pct = relationship_metadata.get('recurring_percentage', 30.0)
        return np.random.random() < (recurring_pct / 100.0)

    def get_typical_amount(self, relationship_metadata: Dict) -> float:
        """
        Get typical transaction amount for institutional relationship

        Args:
            relationship_metadata: Metadata from select_rdfi_for_odfi

        Returns:
            Typical transaction amount (can be used as mean for distribution)
        """
        return relationship_metadata.get('avg_dollar_value', 1000.0)

    def map_routing_to_institution(self, routing_number: str, institutions: List[Dict]) -> Optional[Dict]:
        """
        Map a routing number from the matrix to an institution in the synthetic dataset

        Args:
            routing_number: The routing number from matrix (ODFI or RDFI)
            institutions: List of generated institutions

        Returns:
            Matched institution or None
        """
        # Try exact match first
        for inst in institutions:
            if inst.get('routing_number') == routing_number:
                return inst

        # If no exact match, use deterministic hash to consistently map to same institution
        # This ensures the same routing number always maps to the same synthetic institution
        if institutions:
            hash_value = int(routing_number) if routing_number.isdigit() else hash(routing_number)
            idx = hash_value % len(institutions)
            return institutions[idx]

        return None

    def get_statistics(self) -> Dict:
        """Get statistics about loaded institutional relationships"""
        if not self.is_loaded:
            return {}

        stats = {
            'total_relationships': len(self.matrix_df),
            'unique_institutions': len(self.all_institutions),
            'avg_relationships_per_institution': len(self.matrix_df) / len(self.all_institutions),
            'total_transaction_volume': self.matrix_df['total_dollar_value'].sum() if 'total_dollar_value' in self.matrix_df.columns else 0,
            'total_transaction_count': self.matrix_df['transaction_count'].sum(),
            'avg_relationship_strength': self.matrix_df['relationship_strength'].mean() if 'relationship_strength' in self.matrix_df.columns else 0
        }

        return stats

    def _default_relationship_metadata(self) -> Dict:
        """Return default relationship metadata when no data available"""
        return {
            'transaction_count': 0,
            'total_dollar_value': 0,
            'avg_dollar_value': 1000.0,
            'ach_percentage': 60.0,
            'wire_percentage': 15.0,
            'rtp_percentage': 25.0,
            'recurring_percentage': 30.0,
            'relationship_strength': 0.5
        }

    def create_sample_matrix(self, output_path: str, num_institutions: int = 20):
        """
        Create a sample ODFI-RDFI matrix for testing

        Args:
            output_path: Path to save sample CSV
            num_institutions: Number of institutions to include
        """
        print(f"🔨 Creating sample institutional relationship matrix...")

        # Generate sample routing numbers
        routing_numbers = [f"{i:09d}" for i in range(11000000, 11000000 + num_institutions)]

        # Generate relationships (not all institutions connect to all others)
        relationships = []
        for i, odfi in enumerate(routing_numbers):
            # Each institution connects to 30-70% of other institutions
            num_connections = np.random.randint(int(num_institutions * 0.3), int(num_institutions * 0.7))
            rdfi_candidates = [r for r in routing_numbers if r != odfi]
            rdfi_list = np.random.choice(rdfi_candidates, num_connections, replace=False)

            for rdfi in rdfi_list:
                # Generate realistic transaction patterns
                transaction_count = np.random.lognormal(mean=5, sigma=2)
                transaction_count = max(10, int(transaction_count))

                avg_amount = np.random.lognormal(mean=7, sigma=1.5)
                avg_amount = max(100, avg_amount)

                total_value = transaction_count * avg_amount

                # Payment rail distribution varies by relationship
                ach_pct = np.random.uniform(40, 80)
                wire_pct = np.random.uniform(5, 30)
                rtp_pct = 100 - ach_pct - wire_pct

                # Relationship strength based on volume
                strength = min(1.0, transaction_count / 10000.0)

                relationships.append({
                    'odfi': odfi,
                    'rdfi': rdfi,
                    'transaction_count': transaction_count,
                    'total_dollar_value': round(total_value, 2),
                    'avg_dollar_value': round(avg_amount, 2),
                    'ach_percentage': round(ach_pct, 2),
                    'wire_percentage': round(wire_pct, 2),
                    'rtp_percentage': round(rtp_pct, 2),
                    'recurring_percentage': round(np.random.uniform(10, 50), 2),
                    'avg_settlement_days': round(np.random.uniform(0.5, 2.0), 2),
                    'relationship_strength': round(strength, 4)
                })

        # Create DataFrame and save
        df = pd.DataFrame(relationships)
        df = df.sort_values('transaction_count', ascending=False)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"   ✓ Created {len(relationships)} relationships")
        print(f"   ✓ Saved to {output_path}")

        return df
