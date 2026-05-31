import pandas as pd, numpy as np

old = pd.read_csv(r'data\toronto-dataset\old\KSI.csv', low_memory=False)
new = pd.read_csv(r'data\toronto-dataset\new\Motor Vehicle Collisions with KSI Data - 4326.csv', low_memory=False)

# Parse dates
new['dp'] = pd.to_datetime(new['accdate'], errors='coerce')
new['YEAR'] = new['dp'].dt.year

# Extract ACCNUM from collision_id (format "YEAR:ACCNUM")
new['ACCNUM'] = new['collision_id'].str.split(':', n=1).str[1]

# Keep old as base
base = old.copy()

# Get new records not in old
old_acc = set(base.ACCNUM.astype(str))
new_only = new[~new['ACCNUM'].isin(old_acc)].copy()
print(f'Old KSI: {len(base)} rows, {base.ACCNUM.nunique()} unique ACCNUM')
print(f'New-only: {len(new_only)} rows, {new_only.ACCNUM.nunique()} unique ACCNUM')
print(f'New-only year range: {new_only["YEAR"].min()} - {new_only["YEAR"].max()}')

# Build column mapping: new -> old
col_map = {
    'ACCNUM': 'ACCNUM',
    'accdate': 'DATE',
    'stname1': 'STREET1',
    'stname2': 'STREET2',
    'road_class': 'ROAD_CLASS',
    'accloc': 'ACCLOC',
    'traffictl': 'TRAFFCTL',
    'visible': 'VISIBILITY',
    'light': 'LIGHT',
    'rdsfcond': 'RDSFCOND',
    'acclass': 'ACCLASS',
    'impactype': 'IMPACTYPE',
    'per_inv': 'INVTYPE',
    'invage': 'INVAGE',
    'injury': 'INJURY',
    'fatal_no': 'FATAL_NO',
    'initdir': 'INITDIR',
    'vehtype': 'VEHTYPE',
    'manoeuvre': 'MANOEUVER',
    'drivact': 'DRIVACT',
    'drivcond': 'DRIVCOND',
    'pedtype': 'PEDTYPE',
    'pedact': 'PEDACT',
    'pedcond': 'PEDCOND',
    'cyclistype': 'CYCLISTYPE',
    'cycact': 'CYCACT',
    'cyccond': 'CYCCOND',
    'latitude': 'LATITUDE',
    'longitude': 'LONGITUDE',
    'wardname': 'WARDNUM',
    'division': 'DIVISION',
    'neighbourhood': 'NEIGHBOURHOOD',
    'pedestrian': 'PEDESTRIAN',
    'cyclist': 'CYCLIST',
    'heavy_truck': 'TRUCK',
    'aggressive': 'SPEEDING',
    'red_light': 'REDLIGHT',
}

# Map boolean/int flags
bool_map = {
    'motorcyclist': 'MOTORCYCLE',
    'other_micromobility': 'TRSN_CITY_VEH',
    'older_adult': 'DISABILITY',
    'distracted': 'ALCOHOL',
    'school_child': None,  # No direct mapping
    'failtorem': None,
}

# Copy new-only with column renaming
aligned = new_only.rename(columns=col_map)

# Fill missing columns (present in old but not in new)
for col in base.columns:
    if col not in aligned.columns:
        aligned[col] = np.nan

# Add boolean flags
aligned['AUTOMOBILE'] = ~((aligned['MOTORCYCLE'].fillna(0).astype(int) == 1) | 
                          (aligned['TRUCK'].fillna(0).astype(int) == 1))
aligned['PASSENGER'] = 0  # Default
aligned['CYCLIST'] = 0  # Already have cyclist flag
aligned['PEDESTRIAN'] = 0  # Already have pedestrian flag

# Map new's boolean flags to old format
# New flags are True/False, old uses 'Yes'/'No'
for new_col, old_col in bool_map.items():
    if old_col and new_col in new_only.columns:
        aligned[old_col] = new_only[new_col].map({True: 'Yes', False: 'No'}).fillna('No')

# Fill AUTOMOBILE, EMERG_VEH, AG_DRIV, ALCOHOL, DISABILITY with defaults
aligned['AG_DRIV'] = 'No'
aligned['EMERG_VEH'] = 'No'
if 'ALCOHOL' not in aligned.columns:
    aligned['ALCOHOL'] = 'No'

# For PEDESTRIAN/CYCLIST - use the boolean flags
aligned['PEDESTRIAN'] = new_only['pedestrian'].map({True: 'Yes', False: 'No'}).fillna('No')
aligned['CYCLIST'] = new_only['cyclist'].map({True: 'Yes', False: 'No'}).fillna('No')

# Fill spatial
aligned['X'] = aligned['LONGITUDE']
aligned['Y'] = aligned['LATITUDE']

# Fill identifiers
aligned['INDEX_'] = aligned.index + 1000000
aligned['ObjectId'] = aligned.index + 1000000
aligned['OFFSET'] = 0
aligned['DISTRICT'] = 'Unknown'
aligned['LOCCOORD'] = 'Unknown'
aligned['POLICE_DIVISION'] = aligned['DIVISION']
aligned['HOOD_ID'] = 0
aligned['YEAR'] = new_only['YEAR']

# Parse time from accdate
aligned['TIME'] = new_only['dp'].dt.strftime('%H:%M')
aligned['HOUR'] = new_only['dp'].dt.hour

# Normalize DATE to consistent format (fix mixed timezone-aware/naive parsing)
base['DATE'] = pd.to_datetime(base['DATE'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
aligned['DATE'] = new_only['dp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Reorder columns to match old
aligned = aligned[base.columns]

# Concatenate
merged = pd.concat([base, aligned], ignore_index=True)

# Standardize ACCLASS: new uses 'Fatal Injury', old uses 'Fatal'
merged['ACCLASS'] = merged['ACCLASS'].replace({'Fatal Injury': 'Fatal'})

# Sort by year
merged = merged.sort_values(['YEAR', 'ACCNUM']).reset_index(drop=True)

# Save
out = r'data\toronto-dataset\KSI_merged.csv'
merged.to_csv(out, index=False)
print()
print(f'Merged: {len(merged)} rows, {merged.ACCNUM.nunique()} unique ACCNUM')
print(f'Old KSI: {len(old)} rows')
print(f'Added new-only: {len(aligned)} rows')
print(f'Year range: {merged.YEAR.min()} - {merged.YEAR.max()}')
print(f'Year distribution:')
print(merged.YEAR.value_counts().sort_index().to_string())
print(f'\nACCLASS: {merged.ACCLASS.value_counts().to_dict()}')
