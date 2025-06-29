import chardet
import csv
import pandas as pd

#Adjust num_bytes
def detect_encoding(file_path, num_bytes=10000):
    with open(file_path, 'rb') as f:
        raw = f.read(num_bytes)
    result = chardet.detect(raw)
    print(f"Detected encoding: {result['encoding']} ({100*round(result['confidence'])}% confidence)")
    return result['encoding']

def validate_field_count(file_path, expected_fields, encoding):
    with open(file_path, 'r', encoding=encoding, errors='replace', newline='') as f:
        reader = csv.reader(f, quotechar='"')
        lengths = [len(row) for row in reader]
    unique_lengths = sorted(set(lengths))
    print(f"Unique field counts: {unique_lengths}")
    if expected_fields in unique_lengths and len(unique_lengths) == 1:
        print(f"All rows have {expected_fields} fields.")
    else:
        exit(f"WARNING: Row lengths not consistent.")
    return unique_lengths

def safe_read_csv(file_path, columns, expected_fields):
    try:
        df = pd.read_csv(
            file_path,
            names=columns,
            header=None,
            quotechar='"',
            encoding='utf-8',
            engine='python',
            on_bad_lines='warn',
            encoding_errors='replace'
        )
        encoding_used = 'utf-8'
        print("Loaded file as UTF-8")
    except UnicodeDecodeError:
        encoding_used = detect_encoding(file_path)
        df = pd.read_csv(
            file_path,
            names=columns,
            header=None,
            quotechar='"',
            encoding=encoding_used,
            engine='python',
            on_bad_lines='warn',
            encoding_errors='replace'
        )
        print(f"Loaded file using fallback encoding: {encoding_used}")
    validate_field_count(file_path, expected_fields, encoding_used)
    return df

