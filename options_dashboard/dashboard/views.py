import pandas as pd
from datetime import datetime
import os
import hashlib
import json
from django.http import JsonResponse
from django.core.cache import cache
from .utils import list_option_files, SPOT_CSV
from .greeks import compute_greeks
from .iv import implied_volatility

def get_cache_key(prefix, params):
    """Generate consistent cache key from parameters"""
    param_str = json.dumps(params, sort_keys=True)
    return f"{prefix}_{hashlib.sha256(param_str.encode()).hexdigest()}"

def get_spot_data():
    """Load and cache spot data with mtime-based invalidation"""
    spot_mtime = os.path.getmtime(SPOT_CSV)
    cache_key = f"spot_{spot_mtime}"
    
    if cached := cache.get(cache_key):
        print("[Cache] Spot data loaded from cache")
        return cached
    
    print("[Load] Loading fresh spot data from file")
    spot_df = pd.read_csv(SPOT_CSV, parse_dates=['datetime'])
    cache.set(cache_key, spot_df, timeout=86400)  # 24h cache
    return spot_df

def get_option_data(file_info):
    """Load and cache option data with mtime-based invalidation"""
    file_mtime = os.path.getmtime(file_info['path'])
    cache_key = f"option_{file_mtime}_{file_info['expiry']}_{file_info['strike']}_{file_info['type']}"
    
    if cached := cache.get(cache_key):
        print(f"[Cache] Option data loaded from cache: {file_info['path']}")
        return cached
    
    print(f"[Load] Loading fresh option data from file: {file_info['path']}")
    try:
        option_df = pd.read_csv(file_info['path'], parse_dates=['datetime'])
        option_df = option_df[
            (option_df['strike'] == file_info['strike']) & 
            (option_df['type'] == file_info['type'])
        ]
        cache.set(cache_key, option_df, timeout=86400)  # 24h cache
        return option_df
    except Exception as e:
        print(f"Error loading {file_info['path']}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def filter_by_times(df, time_filters):
    """Filter DataFrame by specified times"""
    if not time_filters or not any(time_filters):
        return df
    
    df = df.copy()
    df['time_str'] = df['datetime'].dt.strftime('%H:%M')
    filtered = df[df['time_str'].isin(time_filters)]
    return filtered.drop(columns=['time_str'])

def merge_with_spot(option_df, spot_df):
    """Merge option data with nearest spot prices"""
    return pd.merge_asof(
        option_df.sort_values('datetime'),
        spot_df[['datetime', 'close']].sort_values('datetime'),
        on='datetime',
        direction='nearest',
        tolerance=pd.Timedelta('1min'),
        suffixes=('', '_spot')
    )

def calculate_all_greeks(merged_df, file_info, params):
    """Vectorized calculation of Greeks for all rows"""
    if merged_df.empty:
        return []
    
    expiry_date = datetime.strptime(file_info['expiry'], "%Y-%m-%d")
    merged_df['T'] = (expiry_date - merged_df['datetime']).dt.total_seconds() / (365 * 24 * 3600)
    valid_rows = merged_df[merged_df['T'] > 0]
    
    if valid_rows.empty:
        return []
    
    results = []
    for _, row in valid_rows.iterrows():
        try:
            iv = implied_volatility(row)  # Assuming this works with row objects
            greeks = compute_greeks(
                S=row['close_spot'],
                K=file_info['strike'],
                T=row['T'],
                r=params['r'],
                sigma=iv,
                option_type=file_info['type']
            )
            results.append({
                'datetime': row['datetime'].isoformat(),
                'expiry': file_info['expiry'],
                'strike': file_info['strike'],
                'type': file_info['type'],
                **greeks
            })
        except Exception as e:
            print(f"Error calculating Greeks: {e}")
            continue
    
    return results

def greeks_view(request):
    """Main view function for Greeks calculation with caching and logging"""
    params = {
        'r': float(request.GET.get('r', 0.05)),
        'time_filter': [
            t for t in request.GET.get('time_filter', '').split(',') 
            if t.strip()
        ] or ['09:15', '10:15', '11:15', '12:15', '13:15', '15:15']
    }
    
    cache_key = get_cache_key('greeks', params)
    
    if cached_result := cache.get(cache_key):
        print("[Cache] Full Greeks result cache hit")
        return JsonResponse(cached_result, safe=False)
    
    print("[Cache Miss] Processing Greeks calculation")
    spot_df = get_spot_data()
    option_files = list_option_files()
    all_results = []
    
    for file_info in option_files:
        option_df = get_option_data(file_info)
        if option_df.empty:
            print(f"[Skip] Empty option data for {file_info['path']}")
            continue
        
        filtered_df = filter_by_times(option_df, params['time_filter'])
        if filtered_df.empty:
            print(f"[Skip] No data after filtering times for {file_info['path']}")
            continue
        
        merged_df = merge_with_spot(filtered_df, spot_df)
        if merged_df.empty:
            print(f"[Skip] Empty merged data for {file_info['path']}")
            continue
        
        file_results = calculate_all_greeks(merged_df, file_info, params)
        all_results.extend(file_results)
    
    if all_results:
        cache.set(cache_key, all_results, timeout=3600)  # Cache final results 1 hour
        print("[Cache] Greeks results cached successfully")
    else:
        print("[Info] No Greeks results generated")
    
    return JsonResponse(all_results, safe=False)
