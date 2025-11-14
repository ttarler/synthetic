"""
Account Behavior Profiles for Realistic Transaction Generation

This module defines realistic account behavior patterns including:
- Transaction frequency profiles
- Amount distribution patterns
- Temporal behavior patterns
- Relationship preferences
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from enum import Enum
from dataclasses import dataclass


class FrequencyTier(Enum):
    """Account activity frequency tiers"""
    DORMANT = "dormant"          # 0-2 transactions per 90 days
    LOW = "low"                  # 3-10 transactions per 90 days
    MEDIUM = "medium"            # 11-25 transactions per 90 days
    HIGH = "high"                # 26-50 transactions per 90 days
    VERY_HIGH = "very_high"      # 51-100 transactions per 90 days


class RelationshipType(Enum):
    """Types of account relationships"""
    RECURRING_PAYMENT = "recurring_payment"    # Regular bills, subscriptions
    PEER_TRANSFER = "peer_transfer"            # Friend/family transfers
    BUSINESS_PAYMENT = "business_payment"      # B2B transactions
    BILL_PAYMENT = "bill_payment"              # Utilities, rent, etc.
    SALARY_DEPOSIT = "salary_deposit"          # Income deposits
    ONE_TIME = "one_time"                      # Ad-hoc transactions


@dataclass
class AccountBehaviorProfile:
    """Complete behavior profile for an account"""
    account_id: str
    account_type: str
    frequency_tier: FrequencyTier
    avg_transactions_per_month: float
    preferred_hours: List[int]           # Hours when account is most active
    preferred_days: List[int]            # Days of week (0=Monday, 6=Sunday)
    amount_mean: float                   # Mean transaction amount
    amount_std: float                    # Std deviation for amounts
    relationship_count: int              # Number of recurring relationships
    intra_institution_preference: float  # 0-1, preference for same institution

    def get_transaction_count(self, days: int) -> int:
        """Get expected number of transactions for a time period"""
        months = days / 30.0
        expected = self.avg_transactions_per_month * months
        # Add some randomness (+/- 20%)
        return int(np.random.normal(expected, expected * 0.2))

    def generate_amount(self) -> float:
        """Generate transaction amount based on profile"""
        # Use log-normal distribution for more realistic amounts
        amount = np.random.lognormal(
            mean=np.log(self.amount_mean),
            sigma=self.amount_std
        )
        # Ensure minimum and reasonable maximum
        return max(10.0, min(amount, self.amount_mean * 10))

    def is_active_hour(self, hour: int) -> bool:
        """Check if hour is within preferred activity window"""
        return hour in self.preferred_hours

    def is_active_day(self, day_of_week: int) -> bool:
        """Check if day is within preferred activity window"""
        return day_of_week in self.preferred_days


class AccountProfileGenerator:
    """Generates realistic account behavior profiles"""

    # Frequency tier distributions by account type
    ACCOUNT_TYPE_FREQUENCY_DIST = {
        'checking': {
            FrequencyTier.DORMANT: 0.10,
            FrequencyTier.LOW: 0.20,
            FrequencyTier.MEDIUM: 0.35,
            FrequencyTier.HIGH: 0.25,
            FrequencyTier.VERY_HIGH: 0.10
        },
        'savings': {
            FrequencyTier.DORMANT: 0.30,
            FrequencyTier.LOW: 0.40,
            FrequencyTier.MEDIUM: 0.20,
            FrequencyTier.HIGH: 0.08,
            FrequencyTier.VERY_HIGH: 0.02
        },
        'business': {
            FrequencyTier.DORMANT: 0.05,
            FrequencyTier.LOW: 0.10,
            FrequencyTier.MEDIUM: 0.25,
            FrequencyTier.HIGH: 0.35,
            FrequencyTier.VERY_HIGH: 0.25
        },
        'trust': {
            FrequencyTier.DORMANT: 0.20,
            FrequencyTier.LOW: 0.35,
            FrequencyTier.MEDIUM: 0.30,
            FrequencyTier.HIGH: 0.12,
            FrequencyTier.VERY_HIGH: 0.03
        }
    }

    # Transaction frequency ranges (per month)
    FREQUENCY_RANGES = {
        FrequencyTier.DORMANT: (0.2, 0.7),      # ~0-2 per 90 days
        FrequencyTier.LOW: (1.0, 3.5),          # ~3-10 per 90 days
        FrequencyTier.MEDIUM: (3.7, 8.5),       # ~11-25 per 90 days
        FrequencyTier.HIGH: (8.7, 16.5),        # ~26-50 per 90 days
        FrequencyTier.VERY_HIGH: (17.0, 35.0)   # ~51-100 per 90 days
    }

    # Amount ranges by account type (mean, std_ratio)
    AMOUNT_PATTERNS = {
        'checking': {
            FrequencyTier.DORMANT: (1500, 0.8),
            FrequencyTier.LOW: (1200, 0.7),
            FrequencyTier.MEDIUM: (800, 0.6),
            FrequencyTier.HIGH: (400, 0.5),
            FrequencyTier.VERY_HIGH: (200, 0.4)
        },
        'savings': {
            FrequencyTier.DORMANT: (5000, 0.9),
            FrequencyTier.LOW: (3500, 0.8),
            FrequencyTier.MEDIUM: (2500, 0.7),
            FrequencyTier.HIGH: (2000, 0.6),
            FrequencyTier.VERY_HIGH: (1500, 0.5)
        },
        'business': {
            FrequencyTier.DORMANT: (8000, 1.0),
            FrequencyTier.LOW: (6000, 0.9),
            FrequencyTier.MEDIUM: (4000, 0.8),
            FrequencyTier.HIGH: (2500, 0.7),
            FrequencyTier.VERY_HIGH: (1500, 0.6)
        },
        'trust': {
            FrequencyTier.DORMANT: (10000, 1.0),
            FrequencyTier.LOW: (7500, 0.9),
            FrequencyTier.MEDIUM: (5000, 0.8),
            FrequencyTier.HIGH: (4000, 0.7),
            FrequencyTier.VERY_HIGH: (3000, 0.6)
        }
    }

    # Relationship count by frequency tier
    RELATIONSHIP_COUNTS = {
        FrequencyTier.DORMANT: (0, 2),
        FrequencyTier.LOW: (1, 4),
        FrequencyTier.MEDIUM: (3, 8),
        FrequencyTier.HIGH: (5, 12),
        FrequencyTier.VERY_HIGH: (8, 20)
    }

    @classmethod
    def generate_profile(cls, account_id: str, account_type: str) -> AccountBehaviorProfile:
        """Generate a realistic behavior profile for an account"""

        # Select frequency tier based on account type
        freq_dist = cls.ACCOUNT_TYPE_FREQUENCY_DIST.get(account_type,
                                                         cls.ACCOUNT_TYPE_FREQUENCY_DIST['checking'])
        frequency_tier = np.random.choice(
            list(freq_dist.keys()),
            p=list(freq_dist.values())
        )

        # Get transaction frequency
        freq_range = cls.FREQUENCY_RANGES[frequency_tier]
        avg_transactions_per_month = np.random.uniform(*freq_range)

        # Get amount patterns
        amount_pattern = cls.AMOUNT_PATTERNS[account_type][frequency_tier]
        amount_mean = amount_pattern[0]
        amount_std = amount_pattern[1]

        # Generate preferred hours (account-specific pattern)
        preferred_hours = cls._generate_preferred_hours(account_type, frequency_tier)

        # Generate preferred days
        preferred_days = cls._generate_preferred_days(account_type)

        # Relationship count
        rel_range = cls.RELATIONSHIP_COUNTS[frequency_tier]
        relationship_count = np.random.randint(*rel_range)

        # Intra-institution preference (higher for personal accounts)
        intra_pref = np.random.beta(5, 2) if account_type in ['checking', 'savings'] else np.random.beta(3, 3)

        return AccountBehaviorProfile(
            account_id=account_id,
            account_type=account_type,
            frequency_tier=frequency_tier,
            avg_transactions_per_month=avg_transactions_per_month,
            preferred_hours=preferred_hours,
            preferred_days=preferred_days,
            amount_mean=amount_mean,
            amount_std=amount_std,
            relationship_count=relationship_count,
            intra_institution_preference=intra_pref
        )

    @classmethod
    def _generate_preferred_hours(cls, account_type: str, frequency_tier: FrequencyTier) -> List[int]:
        """Generate preferred transaction hours based on account characteristics"""
        if account_type == 'business':
            # Business hours (8am-5pm)
            core_hours = list(range(8, 17))
            # Some businesses extend to early morning or evening
            if frequency_tier in [FrequencyTier.HIGH, FrequencyTier.VERY_HIGH]:
                extended = [7] + core_hours + [17, 18]
                return extended
            return core_hours
        elif account_type == 'savings':
            # Savings accounts less time-specific, but favor daytime
            return list(range(7, 20))
        else:
            # Personal accounts - varied patterns
            pattern_type = np.random.choice(['morning', 'lunch', 'evening', 'all_day'])
            if pattern_type == 'morning':
                return list(range(6, 12))
            elif pattern_type == 'lunch':
                return list(range(11, 15))
            elif pattern_type == 'evening':
                return list(range(17, 22))
            else:
                return list(range(7, 21))

    @classmethod
    def _generate_preferred_days(cls, account_type: str) -> List[int]:
        """Generate preferred transaction days (0=Monday, 6=Sunday)"""
        if account_type == 'business':
            # Strongly prefer weekdays
            return list(range(0, 5))  # Monday-Friday
        else:
            # Personal accounts - slight weekday preference but includes weekends
            if np.random.random() < 0.3:
                # Weekend warrior (30% of personal accounts)
                return list(range(0, 7))
            else:
                # Weekday preferred (70%)
                return list(range(0, 6))  # Monday-Saturday


class RelationshipNetwork:
    """Manages account relationships for realistic transaction patterns"""

    def __init__(self):
        self.relationships: Dict[str, List[Dict[str, Any]]] = {}

    def add_relationship(self, sender_id: str, receiver_id: str,
                        relationship_type: RelationshipType, strength: float = 0.5):
        """Add a relationship between two accounts"""
        if sender_id not in self.relationships:
            self.relationships[sender_id] = []

        self.relationships[sender_id].append({
            'receiver_id': receiver_id,
            'type': relationship_type,
            'strength': strength,  # 0-1, higher = more frequent transactions
            'transaction_count': 0
        })

    def get_relationships(self, account_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for an account"""
        return self.relationships.get(account_id, [])

    def select_receiver(self, sender_id: str, use_relationship_prob: float = 0.8) -> Tuple[str, RelationshipType]:
        """Select a receiver account based on relationship network"""
        relationships = self.get_relationships(sender_id)

        if not relationships or np.random.random() > use_relationship_prob:
            # No relationships or random selection
            return None, RelationshipType.ONE_TIME

        # Weight by strength
        strengths = [r['strength'] for r in relationships]
        selected = np.random.choice(relationships, p=np.array(strengths) / sum(strengths))

        # Increment transaction count
        selected['transaction_count'] += 1

        return selected['receiver_id'], selected['type']

    def build_network(self, accounts: List[Dict[str, Any]], profiles: Dict[str, AccountBehaviorProfile]):
        """Build realistic relationship network between accounts"""
        print("🕸️  Building account relationship network...")

        # Create institution-based account groups
        institution_groups = {}
        for account in accounts:
            inst_id = account['institution_id']
            if inst_id not in institution_groups:
                institution_groups[inst_id] = []
            institution_groups[inst_id].append(account)

        # Build relationships for each account
        for account in accounts:
            profile = profiles[account['account_id']]

            # Determine number of relationships to create
            num_relationships = profile.relationship_count

            if num_relationships == 0:
                continue

            # Distribute relationship types
            rel_types = self._distribute_relationship_types(num_relationships, profile.account_type)

            for rel_type in rel_types:
                # Select receiver based on intra-institution preference
                if np.random.random() < profile.intra_institution_preference:
                    # Same institution
                    same_inst_accounts = [a for a in institution_groups[account['institution_id']]
                                         if a['account_id'] != account['account_id']]
                    if same_inst_accounts:
                        receiver = np.random.choice(same_inst_accounts)
                    else:
                        receiver = np.random.choice([a for a in accounts if a['account_id'] != account['account_id']])
                else:
                    # Different institution
                    diff_inst_accounts = [a for a in accounts
                                         if a['account_id'] != account['account_id']
                                         and a['institution_id'] != account['institution_id']]
                    if diff_inst_accounts:
                        receiver = np.random.choice(diff_inst_accounts)
                    else:
                        receiver = np.random.choice([a for a in accounts if a['account_id'] != account['account_id']])

                # Assign strength based on relationship type
                strength = self._get_relationship_strength(rel_type)

                self.add_relationship(
                    account['account_id'],
                    receiver['account_id'],
                    rel_type,
                    strength
                )

        print(f"   Created {sum(len(rels) for rels in self.relationships.values())} relationships")

    def _distribute_relationship_types(self, count: int, account_type: str) -> List[RelationshipType]:
        """Distribute relationship types based on account type"""
        if account_type == 'business':
            # Business accounts favor business payments
            types = [RelationshipType.BUSINESS_PAYMENT] * int(count * 0.6) + \
                   [RelationshipType.RECURRING_PAYMENT] * int(count * 0.3) + \
                   [RelationshipType.BILL_PAYMENT] * int(count * 0.1)
        elif account_type == 'savings':
            # Savings accounts mostly recurring transfers
            types = [RelationshipType.RECURRING_PAYMENT] * int(count * 0.7) + \
                   [RelationshipType.PEER_TRANSFER] * int(count * 0.3)
        else:
            # Personal accounts (checking, trust) - mixed
            types = [RelationshipType.BILL_PAYMENT] * int(count * 0.4) + \
                   [RelationshipType.PEER_TRANSFER] * int(count * 0.3) + \
                   [RelationshipType.RECURRING_PAYMENT] * int(count * 0.2) + \
                   [RelationshipType.BUSINESS_PAYMENT] * int(count * 0.1)

        # Fill remaining with random selection
        while len(types) < count:
            types.append(np.random.choice(list(RelationshipType)))

        return types[:count]

    def _get_relationship_strength(self, rel_type: RelationshipType) -> float:
        """Get relationship strength (transaction frequency) by type"""
        strength_map = {
            RelationshipType.RECURRING_PAYMENT: 0.9,    # Very frequent
            RelationshipType.BILL_PAYMENT: 0.8,         # Frequent
            RelationshipType.SALARY_DEPOSIT: 0.95,      # Very frequent
            RelationshipType.BUSINESS_PAYMENT: 0.7,     # Moderate
            RelationshipType.PEER_TRANSFER: 0.5,        # Less frequent
            RelationshipType.ONE_TIME: 0.1              # Rare
        }
        return strength_map.get(rel_type, 0.5)
