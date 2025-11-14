"""
Field Generator and Validator

Generates and validates field values based on schema field rules configuration.
Supports categories, integers, doubles, strings, timestamps, and derived fields.
"""

import yaml
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import re
try:
    from rstr import xeger  # For regex-based string generation
    HAS_RSTR = True
except ImportError:
    HAS_RSTR = False
    print("⚠️  'rstr' package not installed. Regex string generation will use fallback method.")


class FieldGenerator:
    """Generates field values according to schema field rules"""

    def __init__(self, rules_path: str = 'config/schema_field_rules.yaml'):
        """
        Initialize field generator with rules configuration

        Args:
            rules_path: Path to YAML file containing field rules
        """
        self.rules_path = rules_path
        self.field_rules = self._load_rules()
        self.generation_rules = self.field_rules.get('generation_rules', {})

    def _load_rules(self) -> Dict[str, Any]:
        """Load field rules from YAML configuration"""
        try:
            with open(self.rules_path, 'r') as f:
                rules = yaml.safe_load(f)
            print(f"📋 Loaded field rules from {self.rules_path}")
            return rules
        except FileNotFoundError:
            print(f"⚠️  Field rules file not found: {self.rules_path}")
            return {}
        except Exception as e:
            print(f"⚠️  Error loading field rules: {e}")
            return {}

    def generate_value(self, field_name: str, context: Dict[str, Any] = None) -> Any:
        """
        Generate a value for the specified field according to its rules

        Args:
            field_name: Name of the field
            context: Dictionary of already-generated field values for dependency resolution

        Returns:
            Generated value for the field
        """
        if field_name not in self.field_rules:
            return None

        field_rule = self.field_rules[field_name]
        field_type = field_rule.get('type')
        context = context or {}

        # Handle derived fields first
        if 'derived_from' in field_rule:
            return self._generate_derived_value(field_name, field_rule, context)

        # Handle field mapping (derived from another field's value)
        if 'mapping' in field_rule and context:
            return self._apply_mapping(field_rule, context)

        # Generate based on type
        if field_type == 'category':
            return self._generate_category(field_rule)
        elif field_type == 'integer':
            return self._generate_integer(field_rule)
        elif field_type == 'double':
            return self._generate_double(field_rule)
        elif field_type == 'string':
            return self._generate_string(field_name, field_rule, context)
        elif field_type == 'timestamp':
            return self._generate_timestamp(field_rule, context)
        elif field_type == 'constant':
            return field_rule.get('value', field_rule.get('default', ''))

        # Fallback to default
        return field_rule.get('default', '')

    def _generate_category(self, field_rule: Dict) -> str:
        """Generate categorical value"""
        valid_values = field_rule.get('valid_values', [])
        if not valid_values:
            return field_rule.get('default', '')

        # Use distribution if specified
        distribution = field_rule.get('distribution', {})
        if distribution:
            values = list(distribution.keys())
            probabilities = list(distribution.values())
            # Normalize probabilities
            total = sum(probabilities)
            if total > 0:
                probabilities = [p / total for p in probabilities]
                return np.random.choice(values, p=probabilities)

        # Uniform random selection
        return random.choice(valid_values)

    def _generate_integer(self, field_rule: Dict) -> int:
        """
        Generate integer value with support for different distributions

        Supported distributions:
        - uniform (default): Random integer in range
        - normal: Normal distribution with mean and std
        - poisson: Poisson distribution with lambda
        - binomial: Binomial distribution with n and p
        """
        min_val = field_rule.get('min', 0)
        max_val = field_rule.get('max', 999999999)

        # Handle specific digit requirements
        if 'digits' in field_rule:
            digits = field_rule['digits']
            min_val = 10 ** (digits - 1)
            max_val = (10 ** digits) - 1

        # Check for distribution specification
        distribution = field_rule.get('distribution_type')

        if distribution == 'normal':
            mean = field_rule.get('mean', (min_val + max_val) / 2)
            std = field_rule.get('std', (max_val - min_val) / 6)
            value = int(np.random.normal(mean, std))
            return max(min_val, min(max_val, value))

        elif distribution == 'poisson':
            lam = field_rule.get('lambda', 10)
            value = np.random.poisson(lam)
            return max(min_val, min(max_val, value))

        elif distribution == 'binomial':
            n = field_rule.get('n', 100)
            p = field_rule.get('p', 0.5)
            value = np.random.binomial(n, p)
            return max(min_val, min(max_val, value))

        elif distribution == 'exponential':
            scale = field_rule.get('scale', 1.0)
            value = int(np.random.exponential(scale))
            return max(min_val, min(max_val, value))

        # Default: uniform distribution
        return random.randint(min_val, max_val)

    def _generate_double(self, field_rule: Dict) -> float:
        """
        Generate double/float value with support for different distributions

        Supported distributions:
        - uniform (default): Random float in range
        - normal: Normal distribution with mean and std
        - lognormal: Log-normal distribution with mean and sigma
        - exponential: Exponential distribution with scale
        - beta: Beta distribution with alpha and beta parameters
        - gamma: Gamma distribution with shape and scale
        """
        min_val = field_rule.get('min', 0.0)
        max_val = field_rule.get('max', 999999999.99)
        precision = field_rule.get('precision', 2)

        # Check for distribution specification
        distribution = field_rule.get('distribution_type')

        if distribution == 'normal':
            mean = field_rule.get('mean', (min_val + max_val) / 2)
            std = field_rule.get('std', (max_val - min_val) / 6)
            value = np.random.normal(mean, std)

        elif distribution == 'lognormal':
            mean = field_rule.get('mean', np.log(1000))
            sigma = field_rule.get('sigma', 1.0)
            value = np.random.lognormal(mean, sigma)

        elif distribution == 'exponential':
            scale = field_rule.get('scale', 1.0)
            value = np.random.exponential(scale)

        elif distribution == 'beta':
            alpha = field_rule.get('alpha', 2.0)
            beta_param = field_rule.get('beta', 5.0)
            value = np.random.beta(alpha, beta_param) * (max_val - min_val) + min_val

        elif distribution == 'gamma':
            shape = field_rule.get('shape', 2.0)
            scale = field_rule.get('scale', 2.0)
            value = np.random.gamma(shape, scale)

        elif distribution == 'pareto':
            shape = field_rule.get('shape', 1.5)
            value = (np.random.pareto(shape) + 1) * min_val

        else:
            # Default: uniform distribution
            value = random.uniform(min_val, max_val)

        # Clamp to min/max and round to precision
        value = max(min_val, min(max_val, value))
        return round(value, precision)

    def _generate_string(self, field_name: str, field_rule: Dict, context: Dict) -> str:
        """Generate string value"""
        # Check for regex pattern specification
        if 'pattern' in field_rule and field_rule.get('generate_from_pattern', False):
            pattern = field_rule['pattern']
            if HAS_RSTR:
                try:
                    # Generate string matching the regex pattern
                    return xeger(pattern)
                except Exception as e:
                    print(f"⚠️  Error generating string from pattern {pattern}: {e}")
                    # Fall through to other generation methods
            else:
                # Fallback: generate based on common pattern types
                return self._generate_string_from_pattern_fallback(pattern, field_rule)

        # Check for format specification
        if 'format' in field_rule:
            format_str = field_rule['format']

            # If format references a field from context
            if '{}' in format_str or '{:' in format_str:
                # Try to find referenced field in context
                if 'derived_from' in field_rule:
                    ref_field = field_rule['derived_from']
                    if ref_field in context:
                        return format_str.format(context[ref_field])

                # Generate sequential number if needed
                # This would ideally come from context
                if '{:06d}' in format_str or '{:08d}' in format_str or '{:d}' in format_str:
                    seq_num = context.get('_seq_num', random.randint(0, 999999))
                    return format_str.format(seq_num)

                # Generic format placeholder
                return format_str

            return format_str

        # Use default if specified
        if 'default' in field_rule:
            return field_rule['default']

        # Generate random string if max_length specified
        max_length = field_rule.get('max_length', 50)
        # For name fields, use placeholder
        if 'name' in field_name.lower():
            return f"Entity_{random.randint(1000, 9999)}"

        return ""

    def _generate_string_from_pattern_fallback(self, pattern: str, field_rule: Dict) -> str:
        """
        Fallback method to generate strings from common regex patterns
        when rstr library is not available
        """
        max_length = field_rule.get('max_length', 20)

        # Remove ^ and $ anchors for easier parsing
        clean_pattern = pattern.strip('^$')

        # Handle common specific patterns
        if pattern == r'^0\d{8}$':
            # Fed routing numbers (starts with 0, followed by 8 digits)
            return '0' + ''.join(str(random.randint(0, 9)) for _ in range(8))

        elif pattern == r'^\d{9}$':
            # 9-digit numbers (routing numbers)
            return ''.join(str(random.randint(0, 9)) for _ in range(9))

        elif pattern == r'^[A-Z]{4}US[A-Z0-9]{2}$':
            # SWIFT code pattern
            prefix = ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(4))
            suffix = ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(2))
            return f"{prefix}US{suffix}"

        elif pattern == r'^[A-Z]{3}-\d{6}$':
            # Pattern like ABC-123456
            letters = ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(3))
            digits = ''.join(str(random.randint(0, 9)) for _ in range(6))
            return f"{letters}-{digits}"

        elif pattern == r'^[A-Z]{3,10}$':
            # All uppercase letters, 3-10 chars
            length = random.randint(3, min(10, max_length))
            return ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(length))

        elif pattern == r'^[A-Za-z0-9\s\-]{1,100}$':
            # Alphanumeric with spaces and hyphens
            chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -'
            length = random.randint(5, min(30, max_length))
            return ''.join(random.choice(chars) for _ in range(length))

        # Try to parse pattern components
        # Look for patterns like [A-Z]{3}-\d{6}
        result = []
        i = 0
        while i < len(clean_pattern):
            # Check for character class like [A-Z]
            if clean_pattern[i] == '[':
                end_bracket = clean_pattern.find(']', i)
                if end_bracket > i:
                    char_class = clean_pattern[i:end_bracket+1]
                    i = end_bracket + 1

                    # Check for repetition like {3} or {3,10}
                    count = 1
                    if i < len(clean_pattern) and clean_pattern[i] == '{':
                        end_brace = clean_pattern.find('}', i)
                        if end_brace > i:
                            repetition = clean_pattern[i+1:end_brace]
                            if ',' in repetition:
                                min_rep, max_rep = repetition.split(',')
                                count = random.randint(int(min_rep), int(max_rep))
                            else:
                                count = int(repetition)
                            i = end_brace + 1

                    # Generate characters from class
                    for _ in range(count):
                        result.append(self._generate_from_char_class(char_class))
                else:
                    i += 1

            # Check for \d{n} pattern
            elif i < len(clean_pattern) - 1 and clean_pattern[i:i+2] == r'\d':
                i += 2
                count = 1
                if i < len(clean_pattern) and clean_pattern[i] == '{':
                    end_brace = clean_pattern.find('}', i)
                    if end_brace > i:
                        count = int(clean_pattern[i+1:end_brace])
                        i = end_brace + 1

                result.append(''.join(str(random.randint(0, 9)) for _ in range(count)))

            # Literal character
            else:
                if clean_pattern[i] not in '^$':
                    result.append(clean_pattern[i])
                i += 1

        if result:
            return ''.join(result)

        # Default: generate random alphanumeric string
        chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        length = min(8, max_length)
        return ''.join(random.choice(chars) for _ in range(length))

    def _generate_from_char_class(self, char_class: str) -> str:
        """Generate a random character from a character class like [A-Z] or [0-9a-f]"""
        # Remove brackets
        content = char_class.strip('[]')

        chars = []
        i = 0
        while i < len(content):
            # Check for range like A-Z
            if i < len(content) - 2 and content[i+1] == '-':
                start_char = content[i]
                end_char = content[i+2]
                chars.extend(chr(c) for c in range(ord(start_char), ord(end_char) + 1))
                i += 3
            else:
                chars.append(content[i])
                i += 1

        return random.choice(chars) if chars else 'X'

    def _generate_timestamp(self, field_rule: Dict, context: Dict) -> str:
        """Generate timestamp value"""
        format_str = field_rule.get('format', '%Y-%m-%d %H:%M:%S')

        # Check for derived rule
        derived_rule = field_rule.get('derived_rule', '')

        if 'create_date' in derived_rule and 'transaction_create_date' in context:
            # Derive from creation date with processing delay
            base_date = datetime.strptime(context['transaction_create_date'], format_str)
            if 'processing_delay' in derived_rule:
                delay_hours = random.randint(1, 4)
                result_date = base_date + timedelta(hours=delay_hours)
            else:
                result_date = base_date
            return result_date.strftime(format_str)

        elif 'payment_rail' in derived_rule and context:
            # Settlement date based on payment rail
            payment_rail = context.get('payment_rail', context.get('pay_type', 'ACH'))
            base_date = datetime.strptime(
                context.get('transaction_create_date', datetime.now().strftime(format_str)),
                format_str
            )

            if payment_rail in ['FUNDS', 'RTP', 'CASH']:
                # Instant settlement
                settlement_date = base_date
            elif payment_rail in ['SECURITIES', 'WIRE']:
                # Same day settlement
                settlement_date = base_date + timedelta(hours=2)
            else:  # ACH, CHECK
                # Next business day(s)
                days_to_add = 1 if 'plus_1' not in derived_rule else 2
                settlement_date = base_date + timedelta(days=days_to_add)
                # Skip weekends
                while settlement_date.weekday() >= 5:
                    settlement_date += timedelta(days=1)

            return settlement_date.strftime(format_str)

        # Default: return current timestamp or value from context
        if context and field_rule.get('derived_from') in context:
            return context[field_rule['derived_from']]

        return datetime.now().strftime(format_str)

    def _generate_derived_value(self, field_name: str, field_rule: Dict, context: Dict) -> Any:
        """Generate value derived from another field"""
        derived_from = field_rule['derived_from']

        if derived_from not in context:
            # Return default or empty
            return field_rule.get('default', '')

        source_value = context[derived_from]

        # Apply format if specified
        if 'format' in field_rule:
            format_str = field_rule['format']
            return format_str.format(source_value)

        # Direct copy
        return source_value

    def _apply_mapping(self, field_rule: Dict, context: Dict) -> Any:
        """Apply value mapping from context field"""
        mapping = field_rule.get('mapping', {})

        # Find the source field (usually payment_rail or pay_type)
        source_field = None
        for possible_source in ['payment_rail', 'pay_type', 'originating_account_type']:
            if possible_source in context:
                source_field = possible_source
                break

        if not source_field:
            return field_rule.get('default', self._generate_category(field_rule))

        source_value = context[source_field]

        # Return mapped value or default
        return mapping.get(source_value, field_rule.get('default', ''))

    def apply_dependencies(self, field_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply cross-field dependencies and rules

        Args:
            field_values: Dictionary of already-generated field values

        Returns:
            Updated field values with dependencies applied
        """
        dependencies = self.generation_rules.get('dependencies', [])

        for dep in dependencies:
            if_field = dep.get('if_field')
            equals = dep.get('equals')
            then_set = dep.get('then_set', {})

            if if_field in field_values and field_values[if_field] == equals:
                for target_field, target_value in then_set.items():
                    if target_value == "random_from_distribution":
                        # Generate random value for this field
                        if target_field in self.field_rules:
                            field_values[target_field] = self.generate_value(target_field, field_values)
                    else:
                        field_values[target_field] = target_value

        # Apply same-institution vs cross-institution rules
        if 'odfi' in field_values and 'rdfi' in field_values:
            if field_values['odfi'] == field_values['rdfi']:
                # Same institution
                same_inst_rules = self.generation_rules.get('same_institution_rules', {})
                field_values.update(same_inst_rules)
            else:
                # Cross institution
                cross_inst_rules = self.generation_rules.get('cross_institution_rules', {})
                field_values.update(cross_inst_rules)

        return field_values

    def validate_value(self, field_name: str, value: Any) -> tuple[bool, str]:
        """
        Validate a field value against its rules

        Args:
            field_name: Name of the field
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if field_name not in self.field_rules:
            return True, ""  # No rules to validate against

        field_rule = self.field_rules[field_name]

        # Check required
        if field_rule.get('required', False) and (value is None or value == ''):
            return False, f"{field_name} is required but empty"

        # Type-specific validation
        field_type = field_rule.get('type')

        if field_type == 'category':
            valid_values = field_rule.get('valid_values', [])
            if valid_values and value not in valid_values:
                return False, f"{field_name} value '{value}' not in valid values: {valid_values}"

        elif field_type == 'integer':
            if not isinstance(value, int):
                return False, f"{field_name} must be integer, got {type(value)}"
            min_val = field_rule.get('min')
            max_val = field_rule.get('max')
            if min_val is not None and value < min_val:
                return False, f"{field_name} value {value} below minimum {min_val}"
            if max_val is not None and value > max_val:
                return False, f"{field_name} value {value} above maximum {max_val}"

        elif field_type == 'double':
            if not isinstance(value, (int, float)):
                return False, f"{field_name} must be numeric, got {type(value)}"
            min_val = field_rule.get('min')
            max_val = field_rule.get('max')
            if min_val is not None and value < min_val:
                return False, f"{field_name} value {value} below minimum {min_val}"
            if max_val is not None and value > max_val:
                return False, f"{field_name} value {value} above maximum {max_val}"

        elif field_type == 'string':
            if not isinstance(value, str):
                return False, f"{field_name} must be string, got {type(value)}"
            max_length = field_rule.get('max_length')
            if max_length and len(value) > max_length:
                return False, f"{field_name} length {len(value)} exceeds maximum {max_length}"
            pattern = field_rule.get('pattern')
            if pattern and not re.match(pattern, value):
                return False, f"{field_name} value '{value}' does not match pattern {pattern}"

        return True, ""

    def get_field_info(self, field_name: str) -> Dict[str, Any]:
        """Get rule information for a field"""
        return self.field_rules.get(field_name, {})

    def list_all_fields(self) -> List[str]:
        """Get list of all configured fields"""
        return [k for k in self.field_rules.keys() if k != 'generation_rules']
