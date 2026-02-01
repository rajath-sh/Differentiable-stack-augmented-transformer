"""
Data generators for Dyck language and Arithmetic expressions.
"""

import random


# ==============================================================================
# DYCK LANGUAGE (BALANCED PARENTHESES)
# ==============================================================================

def is_valid_brackets(seq):
    """Check if a bracket sequence is valid (balanced)."""
    depth = 0
    for c in seq:
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def generate_valid_dyck(length):
    """Generate a valid (balanced) bracket sequence of given length."""
    if length % 2 == 1:
        length += 1
    
    seq = []
    opens = 0
    
    for pos in range(length):
        remaining = length - pos
        # Must close if we have as many opens as remaining positions
        if remaining == opens:
            seq.append(')')
            opens -= 1
        # Can open if we have room
        elif opens == 0:
            seq.append('(')
            opens += 1
        # Random choice otherwise
        elif random.random() < 0.5:
            seq.append('(')
            opens += 1
        else:
            seq.append(')')
            opens -= 1
    
    return ''.join(seq)


def generate_invalid_dyck(length):
    """Generate an invalid bracket sequence by corrupting a valid one."""
    if length % 2 == 1:
        length += 1
    
    seq = list(generate_valid_dyck(length))
    
    # Try multiple corruption strategies
    max_attempts = 10
    for _ in range(max_attempts):
        corrupted = seq.copy()
        
        # Strategy: swap 2-4 random positions
        num_swaps = random.randint(2, min(4, len(corrupted) // 2))
        indices = random.sample(range(len(corrupted)), num_swaps * 2)
        for i in range(0, len(indices), 2):
            corrupted[indices[i]], corrupted[indices[i+1]] = \
                corrupted[indices[i+1]], corrupted[indices[i]]
        
        result = ''.join(corrupted)
        if not is_valid_brackets(result):
            return result
    
    # Fallback: flip a character
    idx = random.randint(0, len(seq) - 1)
    seq[idx] = ')' if seq[idx] == '(' else '('
    return ''.join(seq)


def generate_dyck_data(n_samples, length_range, balanced_ratio=0.5):
    """
    Generate Dyck language dataset.
    
    Args:
        n_samples: Total number of samples
        length_range: (min_length, max_length) tuple
        balanced_ratio: Ratio of valid to invalid samples
    
    Returns:
        sequences: List of bracket strings
        labels: List of labels (0=valid, 1=invalid)
    """
    sequences = []
    labels = []
    
    min_len, max_len = length_range
    n_valid = int(n_samples * balanced_ratio)
    n_invalid = n_samples - n_valid
    
    # Generate valid sequences
    for _ in range(n_valid):
        length = random.randint(min_len, max_len)
        # Make even
        if length % 2 == 1:
            length += 1
        sequences.append(generate_valid_dyck(length))
        labels.append(0)  # Valid = 0
    
    # Generate invalid sequences
    for _ in range(n_invalid):
        length = random.randint(min_len, max_len)
        if length % 2 == 1:
            length += 1
        sequences.append(generate_invalid_dyck(length))
        labels.append(1)  # Invalid = 1
    
    # Shuffle
    combined = list(zip(sequences, labels))
    random.shuffle(combined)
    sequences, labels = zip(*combined)
    
    return list(sequences), list(labels)


# ==============================================================================
# DYCK-2 LANGUAGE (MULTI-BRACKET TYPES)
# ==============================================================================
# Dyck-2 uses two bracket types: ( ) and [ ]
# This is HARDER because the stack must track which TYPE of bracket to close

BRACKET_PAIRS = [('(', ')'), ('[', ']')]


def is_valid_dyck2(seq):
    """Check if a multi-bracket sequence is valid."""
    stack = []
    matching = {')': '(', ']': '['}
    
    for c in seq:
        if c in '([':
            stack.append(c)
        elif c in ')]':
            if not stack or stack[-1] != matching[c]:
                return False
            stack.pop()
    
    return len(stack) == 0


def generate_valid_dyck2(length):
    """Generate a valid multi-bracket sequence of given length."""
    if length % 2 == 1:
        length += 1
    
    seq = []
    stack = []  # Track open brackets
    
    for pos in range(length):
        remaining = length - pos
        
        # Must close if we have as many opens as remaining
        if remaining == len(stack):
            open_b = stack.pop()
            close_b = ')' if open_b == '(' else ']'
            seq.append(close_b)
        # Can open if we have room
        elif len(stack) == 0:
            open_b, close_b = random.choice(BRACKET_PAIRS)
            seq.append(open_b)
            stack.append(open_b)
        # Random choice
        elif random.random() < 0.5:
            open_b, close_b = random.choice(BRACKET_PAIRS)
            seq.append(open_b)
            stack.append(open_b)
        else:
            open_b = stack.pop()
            close_b = ')' if open_b == '(' else ']'
            seq.append(close_b)
    
    return ''.join(seq)


def generate_invalid_dyck2(length):
    """Generate an invalid multi-bracket sequence by corrupting a valid one."""
    if length % 2 == 1:
        length += 1
    
    seq = list(generate_valid_dyck2(length))
    
    max_attempts = 10
    for _ in range(max_attempts):
        corrupted = seq.copy()
        
        # Strategy 1: Swap bracket types (most effective for Dyck-2)
        # Find a ( or [ and change it to the other type
        bracket_indices = [i for i, c in enumerate(corrupted) if c in '([']
        if len(bracket_indices) >= 1:
            idx = random.choice(bracket_indices)
            # Swap bracket type
            corrupted[idx] = '[' if corrupted[idx] == '(' else '('
        
        result = ''.join(corrupted)
        if not is_valid_dyck2(result):
            return result
        
        # Strategy 2: Swap two positions
        if len(corrupted) >= 4:
            i, j = random.sample(range(len(corrupted)), 2)
            corrupted[i], corrupted[j] = corrupted[j], corrupted[i]
            result = ''.join(corrupted)
            if not is_valid_dyck2(result):
                return result
    
    # Fallback: swap mismatched brackets
    idx = random.randint(0, len(seq) - 1)
    if seq[idx] == '(':
        seq[idx] = ']'  # Intentional mismatch
    elif seq[idx] == '[':
        seq[idx] = ')'
    elif seq[idx] == ')':
        seq[idx] = '['
    else:
        seq[idx] = '('
    
    return ''.join(seq)


def generate_dyck2_data(n_samples, length_range, balanced_ratio=0.5):
    """
    Generate Dyck-2 (multi-bracket) dataset.
    
    This is HARDER than Dyck-1 because the model must track:
    - Stack depth (like Dyck-1)
    - Bracket TYPE (which Dyck-1 doesn't require)
    
    Args:
        n_samples: Total number of samples
        length_range: (min_length, max_length) tuple
        balanced_ratio: Ratio of valid to invalid samples
    
    Returns:
        sequences: List of multi-bracket strings
        labels: List of labels (0=valid, 1=invalid)
    """
    sequences = []
    labels = []
    
    min_len, max_len = length_range
    n_valid = int(n_samples * balanced_ratio)
    n_invalid = n_samples - n_valid
    
    # Generate valid sequences
    for _ in range(n_valid):
        length = random.randint(min_len, max_len)
        if length % 2 == 1:
            length += 1
        sequences.append(generate_valid_dyck2(length))
        labels.append(0)
    
    # Generate invalid sequences
    for _ in range(n_invalid):
        length = random.randint(min_len, max_len)
        if length % 2 == 1:
            length += 1
        sequences.append(generate_invalid_dyck2(length))
        labels.append(1)
    
    # Shuffle
    combined = list(zip(sequences, labels))
    random.shuffle(combined)
    sequences, labels = zip(*combined)
    
    return list(sequences), list(labels)


# ==============================================================================
# ARITHMETIC EXPRESSIONS
# ==============================================================================

OPERATORS = ['+', '-', '*']


def generate_arithmetic_expr(depth=0, max_depth=3):
    """Generate a random nested arithmetic expression."""
    if depth >= max_depth or random.random() < 0.4:
        return str(random.randint(1, 9))
    
    left = generate_arithmetic_expr(depth + 1, max_depth)
    right = generate_arithmetic_expr(depth + 1, max_depth)
    op = random.choice(OPERATORS)
    
    return f"({left} {op} {right})"


def safe_eval(expr):
    """Safely evaluate an arithmetic expression."""
    try:
        return eval(expr)
    except Exception:
        return None


def corrupt_arithmetic_expr(expr):
    """
    Corrupt an arithmetic expression to make it evaluate to a different result.
    We change the claimed result, not the expression itself.
    """
    result = safe_eval(expr)
    if result is None:
        return None, None
    
    # Generate an incorrect result
    if random.random() < 0.5:
        # Add or subtract a random amount
        wrong_result = result + random.choice([-2, -1, 1, 2, 3, 5, 10])
    else:
        # Multiply or divide
        wrong_result = result * random.choice([2, -1]) if result != 0 else random.randint(1, 10)
    
    return expr, int(wrong_result)


def generate_arithmetic_data(n_samples, depth_range, balanced_ratio=0.5):
    """
    Generate arithmetic dataset.
    Each sample is (expression, claimed_result) and label indicates if result is correct.
    
    Args:
        n_samples: Total number of samples
        depth_range: (min_depth, max_depth) tuple for expression nesting
        balanced_ratio: Ratio of correct to incorrect samples
    
    Returns:
        sequences: List of "expr = result" strings
        labels: List of labels (0=correct, 1=incorrect)
    """
    sequences = []
    labels = []
    
    min_depth, max_depth = depth_range
    n_correct = int(n_samples * balanced_ratio)
    n_incorrect = n_samples - n_correct
    
    # Generate correct evaluations
    for _ in range(n_correct):
        depth = random.randint(min_depth, max_depth)
        expr = generate_arithmetic_expr(0, depth)
        result = safe_eval(expr)
        if result is not None:
            sequences.append(f"{expr} = {int(result)}")
            labels.append(0)  # Correct = 0
    
    # Generate incorrect evaluations
    attempts = 0
    while len(labels) - n_correct < n_incorrect and attempts < n_incorrect * 2:
        attempts += 1
        depth = random.randint(min_depth, max_depth)
        expr = generate_arithmetic_expr(0, depth)
        result = safe_eval(expr)
        if result is not None:
            expr, wrong_result = corrupt_arithmetic_expr(expr)
            if wrong_result is not None and wrong_result != result:
                sequences.append(f"{expr} = {wrong_result}")
                labels.append(1)  # Incorrect = 1
    
    # Shuffle
    combined = list(zip(sequences, labels))
    random.shuffle(combined)
    sequences, labels = zip(*combined)
    
    return list(sequences), list(labels)


# ==============================================================================
# COMBINED DATA GENERATION
# ==============================================================================

def generate_combined_data(dyck_samples, arithmetic_samples, length_range, depth_range):
    """Generate combined dataset from both tasks."""
    # Generate each task's data
    dyck_seqs, dyck_labels = generate_dyck_data(dyck_samples, length_range)
    arith_seqs, arith_labels = generate_arithmetic_data(arithmetic_samples, depth_range)
    
    # Combine
    all_seqs = dyck_seqs + arith_seqs
    all_labels = dyck_labels + arith_labels
    
    # Shuffle
    combined = list(zip(all_seqs, all_labels))
    random.shuffle(combined)
    all_seqs, all_labels = zip(*combined)
    
    return list(all_seqs), list(all_labels)
