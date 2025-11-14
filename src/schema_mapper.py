"""
Schema Mapper for Transaction Data

Maps simplified transaction data to the full payment system schema
as defined in config/schema.txt. Generates realistic payment rail-specific
fields including ODFI/RDFI information, settlement dates, and payment metadata.

Now integrated with FieldGenerator for rule-based field generation.
"""

from datetime import datetime, timedelta
from typing import Dict, Any
import random
import hashlib
from src.field_generator import FieldGenerator


class PaymentSchemaMapper:
    """Maps transactions to full payment system schema"""

    # Routing number prefixes by institution type
    ROUTING_PREFIXES = {
        'major_bank': ['011', '021', '026', '031', '041', '051', '061', '071'],
        'regional_bank': ['022', '032', '042', '052', '062', '072', '082'],
        'community_bank': ['023', '033', '043', '053', '063', '073', '083'],
        'credit_union': '025',
        'fintech': ['084', '094']
    }

    # Payment type mappings
    PAYMENT_TYPE_DETAILS = {
        'ACH': {
            'pay_type': 'ACH',
            'pay_code': 'CCD',  # Corporate Credit or Debit
            'pay_code_type': 'SEC_CODE',
            'payment_speed_category': 'STANDARD'
        },
        'WIRE': {
            'pay_type': 'WIRE',
            'pay_code': 'CTR',  # Customer Transfer
            'pay_code_type': 'TYPE_CODE',
            'payment_speed_category': 'EXPEDITED'
        },
        'RTP': {
            'pay_type': 'RTP',
            'pay_code': 'P2P',  # Person to Person
            'pay_code_type': 'REQUEST_TYPE',
            'payment_speed_category': 'INSTANT'
        }
    }

    # Account type mappings
    ACCOUNT_TYPE_MAPPINGS = {
        'checking': {
            'account_type': 'DDA',  # Demand Deposit Account
            'description': 'Personal Checking'
        },
        'savings': {
            'account_type': 'SAV',  # Savings
            'description': 'Personal Savings'
        },
        'business': {
            'account_type': 'DDA',
            'description': 'Business Checking'
        },
        'trust': {
            'account_type': 'TRU',  # Trust
            'description': 'Trust Account'
        }
    }

    # FedLine tiers
    FEDLINE_TIERS = ['TIER_1', 'TIER_2', 'TIER_3']

    def __init__(self, institutions: list, field_rules_path: str = 'config/schema_field_rules.yaml'):
        """
        Initialize schema mapper with institution data

        Args:
            institutions: List of institution dictionaries with id, type, name
            field_rules_path: Path to field rules configuration file
        """
        self.institutions = {inst['institution_id']: inst for inst in institutions}
        self.field_generator = FieldGenerator(field_rules_path)
        self._generate_institution_codes()

    def _generate_institution_codes(self):
        """Generate routing numbers and identifiers for institutions"""
        for inst_id, inst in self.institutions.items():
            # Skip if routing number already exists (e.g., from institutional relationship matrix)
            if 'routing_number' in inst and inst['routing_number']:
                # Still generate SWIFT code and fedline tier if missing
                if 'swift_code' not in inst:
                    inst['swift_code'] = inst['name'].replace(' ', '')[:4].upper() + 'US' + inst_id[-2:]
                if 'fedline_tier' not in inst:
                    # Use field generator for fedline tier
                    inst['fedline_tier'] = self.field_generator.generate_value('sending_fedline_tier')
                continue

            inst_type = inst.get('type', 'community_bank')

            # Generate routing number using field generator (9-digit integer)
            routing = self.field_generator.generate_value('odfi')

            # Generate SWIFT-like code (8 chars)
            swift_code = inst['name'].replace(' ', '')[:4].upper() + 'US' + inst_id[-2:]

            inst['routing_number'] = routing
            inst['swift_code'] = swift_code
            inst['fedline_tier'] = self.field_generator.generate_value('sending_fedline_tier')

    def map_to_schema(self, transaction: Dict[str, Any], accounts: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Map simplified transaction to full payment schema using FieldGenerator

        Args:
            transaction: Simplified transaction dict with basic fields
            accounts: Dictionary mapping account_id to account details

        Returns:
            Full schema-compliant transaction dictionary
        """
        # Get sender and receiver details
        sender_account = accounts[transaction['sender_account']]
        receiver_account = accounts[transaction['receiver_account']]
        sender_inst = self.institutions[transaction['sender_institution']]
        receiver_inst = self.institutions[transaction['receiver_institution']]

        # Build context dictionary for field generation
        context = {}

        # Preserve or generate payment_rail
        # If transaction already has payment_rail (from generator), use it
        # Otherwise map from payment_type
        if 'payment_rail' in transaction:
            payment_rail = transaction['payment_rail']
        else:
            payment_type = transaction.get('payment_type', 'ACH')
            payment_rail_map = {'ACH': 'ACH', 'WIRE': 'SECURITIES', 'RTP': 'FUNDS'}
            payment_rail = payment_rail_map.get(payment_type, 'ACH')
        context['payment_rail'] = payment_rail

        # Keep payment_type for backward compatibility with settlement logic
        payment_type = transaction.get('payment_type', 'ACH')

        # Generate transaction identifiers
        tx_id_1 = transaction['transaction_id']
        context['transaction_id_1'] = tx_id_1
        context['transaction_id_2'] = self.field_generator.generate_value('transaction_id_2', context)

        # Institution details (preserve from matrix/existing data)
        context['odfi'] = sender_inst['routing_number']
        context['odfi_name'] = sender_inst['name']
        context['rdfi'] = receiver_inst['routing_number']
        context['rdfi_name'] = receiver_inst['name']

        # FedLine tiers (already generated in institution codes)
        context['sending_fedline_tier'] = sender_inst['fedline_tier']
        context['receiving_fedline_tier'] = receiver_inst['fedline_tier']

        # Network sources (mapped from payment_rail)
        context['sending_net_source'] = self.field_generator.generate_value('sending_net_source', context)
        context['receiving_net_source'] = self.field_generator.generate_value('receiving_net_source', context)

        # Instructing FI details
        context['instructing_fi_id_code'] = self.field_generator.generate_value('instructing_fi_id_code', context)
        context['instructing_fi_id'] = sender_inst.get('swift_code', sender_inst['routing_number'])
        context['instructing_fi_name'] = sender_inst['name']

        # Originating FI details (typically same as ODFI)
        context['ofi_id_code'] = self.field_generator.generate_value('ofi_id_code', context)
        context['ofi_id'] = str(sender_inst['routing_number'])
        context['ofi_name'] = sender_inst['name']

        # Originating account details
        context['originating_account_type'] = self._get_account_type(sender_account['account_type'])
        context['originating_account_identifier'] = sender_account['account_id']
        context['originating_account_description'] = self._get_account_description(sender_account['account_type'])

        # Intermediary FI (for cross-institution transactions)
        same_institution = sender_inst['institution_id'] == receiver_inst['institution_id']
        if not same_institution:
            context['intermediary_fi_id_code'] = 'ROUTING'
            context['intermediary_fi_id'] = self._get_intermediary_routing()
            context['intermediary_fi_name'] = 'Federal Reserve Bank'
        else:
            context['intermediary_fi_id_code'] = ''
            context['intermediary_fi_id'] = ''
            context['intermediary_fi_name'] = ''

        # Receiving FI details
        context['rfi_id_code'] = self.field_generator.generate_value('rfi_id_code', context)
        context['rfi_id'] = str(receiver_inst['routing_number'])
        context['rfi_name'] = receiver_inst['name']

        # Receiving account details
        context['receiving_account_type'] = self._get_account_type(receiver_account['account_type'])
        context['receiving_account_identifier'] = receiver_account['account_id']
        context['receiving_account_description'] = self._get_account_description(receiver_account['account_type'])

        # Payment type details (use field generator with mappings)
        context['pay_type'] = self.field_generator.generate_value('pay_type', context)

        # Determine pay subtypes
        pay_subtypes = self._determine_pay_subtypes(transaction, sender_account, receiver_account)
        context['pay_type_subtype_1'] = pay_subtypes[0]
        context['pay_type_subtype_2'] = pay_subtypes[1]

        context['return_/_error_reason'] = ''  # Empty for successful transactions
        context['pay_code'] = self.field_generator.generate_value('pay_code', context)
        context['pay_code_type'] = self.field_generator.generate_value('pay_code_type', context)

        # Transaction metadata
        context['originator_type'] = self._determine_originator_type(sender_account)
        context['transaction_purpose'] = self._determine_transaction_purpose(transaction)
        context['value_type'] = self.field_generator.generate_value('value_type', context)
        context['reference_id'] = f"REF_{tx_id_1}"
        context['previous_message_id'] = ''
        context['payment_speed_category'] = self.field_generator.generate_value('payment_speed_category', context)

        # Timestamps - use provided timestamp and derive others
        timestamp = transaction['timestamp']
        context['transaction_create_date'] = timestamp

        # Calculate settlement dates
        timestamp_dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        settlement_dates = self._calculate_settlement_dates(timestamp_dt, payment_type, same_institution)
        context['transaction_processed_date'] = settlement_dates['processed'].strftime('%Y-%m-%d %H:%M:%S')
        context['odfi_transaction_settlement_date'] = settlement_dates['odfi_settlement'].strftime('%Y-%m-%d %H:%M:%S')
        context['rdfi_transaction_settlement_date'] = settlement_dates['rdfi_settlement'].strftime('%Y-%m-%d %H:%M:%S')

        # Amounts and directions
        context['dollar_value'] = transaction['amount']
        context['security_par_value'] = 0.0
        context['dollar_direction'] = self.field_generator.generate_value('dollar_direction', context)
        context['security_direction'] = ''

        # Additional fields
        context['addenda_count'] = self.field_generator.generate_value('addenda_count', context)
        context['recurring_payment'] = self._is_recurring_payment(transaction)

        # Apply cross-field dependencies
        context = self.field_generator.apply_dependencies(context)

        # Preserve fraud-related fields
        context['is_fraud'] = transaction.get('is_fraud', False)
        context['fraud_type'] = transaction.get('fraud_type', '')
        context['risk_score'] = transaction.get('risk_score', 0)
        context['relationship_type'] = transaction.get('relationship_type', '')

        return context

    def _calculate_settlement_dates(self, transaction_date: datetime,
                                    payment_type: str, same_institution: bool) -> Dict[str, datetime]:
        """Calculate realistic settlement dates based on payment type"""
        if payment_type == 'RTP':
            # Real-time payments settle immediately
            return {
                'processed': transaction_date,
                'odfi_settlement': transaction_date,
                'rdfi_settlement': transaction_date
            }
        elif payment_type == 'WIRE':
            # Wire transfers process same day
            processed = transaction_date
            settlement = transaction_date + timedelta(hours=2)
            return {
                'processed': processed,
                'odfi_settlement': settlement,
                'rdfi_settlement': settlement
            }
        else:  # ACH
            # ACH typically settles in 1-2 business days
            processed = transaction_date + timedelta(hours=4)
            settlement_days = 0 if same_institution else random.choice([1, 2])
            odfi_settlement = self._next_business_day(transaction_date, settlement_days)
            rdfi_settlement = self._next_business_day(transaction_date, settlement_days + 1)
            return {
                'processed': processed,
                'odfi_settlement': odfi_settlement,
                'rdfi_settlement': rdfi_settlement
            }

    def _next_business_day(self, date: datetime, days_ahead: int = 0) -> datetime:
        """Get next business day, skipping weekends"""
        current = date + timedelta(days=days_ahead)
        while current.weekday() >= 5:  # Saturday=5, Sunday=6
            current += timedelta(days=1)
        return current

    def _get_account_type(self, account_type_code: str) -> str:
        """Get formal account type from code"""
        return self.ACCOUNT_TYPE_MAPPINGS.get(account_type_code, {}).get('account_type', 'DDA')

    def _get_account_description(self, account_type_code: str) -> str:
        """Get account description from code"""
        return self.ACCOUNT_TYPE_MAPPINGS.get(account_type_code, {}).get('description', 'Checking Account')

    def _get_intermediary_routing(self) -> str:
        """Get Federal Reserve intermediary routing number"""
        # Fed routing numbers start with 0 followed by district (1-12)
        district = random.randint(1, 12)
        return f"0{district:02d}000000"

    def _determine_originator_type(self, account: Dict) -> str:
        """Determine originator type based on account"""
        if account['account_type'] == 'business':
            return 'CORPORATE'
        elif account['account_type'] == 'trust':
            return 'FIDUCIARY'
        else:
            return 'INDIVIDUAL'

    def _determine_transaction_purpose(self, transaction: Dict) -> str:
        """Determine transaction purpose based on relationship type"""
        rel_type = transaction.get('relationship_type', 'one_time')
        purpose_map = {
            'bill_payment': 'BILL_PAYMENT',
            'recurring_payment': 'SUBSCRIPTION',
            'salary_deposit': 'PAYROLL',
            'peer_transfer': 'PERSON_TO_PERSON',
            'business_payment': 'TRADE_SETTLEMENT',
            'one_time': 'GENERAL_PAYMENT'
        }
        return purpose_map.get(rel_type, 'GENERAL_PAYMENT')

    def _determine_pay_subtypes(self, transaction: Dict,
                                sender_account: Dict, receiver_account: Dict) -> tuple:
        """Determine payment subtypes based on transaction characteristics"""
        rel_type = transaction.get('relationship_type', 'one_time')
        amount = transaction['amount']

        # First subtype based on relationship
        if rel_type == 'bill_payment':
            subtype1 = 'UTILITY'
        elif rel_type == 'salary_deposit':
            subtype1 = 'SALARY'
        elif rel_type == 'business_payment':
            subtype1 = 'INVOICE'
        elif rel_type == 'peer_transfer':
            subtype1 = 'PERSONAL'
        else:
            subtype1 = 'STANDARD'

        # Second subtype based on amount and account types
        if amount > 10000:
            subtype2 = 'HIGH_VALUE'
        elif sender_account['account_type'] == 'business':
            subtype2 = 'COMMERCIAL'
        else:
            subtype2 = 'CONSUMER'

        return (subtype1, subtype2)

    def _is_recurring_payment(self, transaction: Dict) -> str:
        """Determine if transaction is recurring"""
        rel_type = transaction.get('relationship_type', 'one_time')
        if rel_type in ['recurring_payment', 'bill_payment', 'salary_deposit']:
            return 'YES'
        return 'NO'
