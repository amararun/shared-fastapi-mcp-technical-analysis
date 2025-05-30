# Chart Scaling Analysis & Implementation Guide

## Overview
This FastAPI Technical Analysis application implements sophisticated chart scaling mechanisms using **matplotlib** to create proportionally scaled, high-quality financial charts. The scaling system ensures charts maintain proper aspect ratios, readability, and visual consistency across different data ranges and timeframes.

## Key Scaling Mechanisms Identified

### 1. **Figure Size Scaling (`figsize` parameter)**
The app uses strategic figure sizing for different chart types:

```python
# Single technical chart - 12x12 inches for square proportions
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), 
                                   height_ratios=[2, 1, 1], 
                                   sharex=True, 
                                   gridspec_kw={'hspace': 0})

# Combined side-by-side charts - 24x12 inches for wide layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
```

**Key Benefits:**
- Maintains 1:1 aspect ratio for individual charts
- 2:1 aspect ratio for combined charts provides optimal viewing
- Large figure size ensures high resolution and readability

### 2. **Volume Normalization & Proportional Scaling**
The most sophisticated scaling mechanism handles volume bars relative to price data:

```python
# Calculate bar width based on date range - CRITICAL for proportions
bar_width = (df['DATE'].iloc[-1] - df['DATE'].iloc[0]).days / len(df) * 0.8
if bar_width <= 0:
    bar_width = 0.8  # Default width if calculation fails

# Normalize volume to make it visible without overwhelming price data
price_range = df['CLOSE'].max() - df['CLOSE'].min()
volume_scale_factor = price_range * 0.2 / df['VOLUME'].max() if df['VOLUME'].max() > 0 else 0.2
normalized_volume = df['VOLUME'] * volume_scale_factor

# Apply scaling to volume axis - limits volume to 30% of price range
ax1v.set_ylim(0, price_range * 0.3)
```

**Key Benefits:**
- Volume bars scale proportionally to the price range
- Volume never overwhelms the main price chart (limited to 30% of price range)
- Bar width adapts to data density

### 3. **Subplot Height Ratios**
Strategic allocation of vertical space:

```python
height_ratios=[2, 1, 1]  # Price gets 50%, MACD and RSI each get 25%
```

### 4. **DPI and Export Scaling**
High-quality image export with consistent scaling:

```python
fig.savefig(temp_path, dpi=150, bbox_inches='tight')
```

**Key Benefits:**
- `dpi=150` ensures crisp, high-resolution output
- `bbox_inches='tight'` removes excess whitespace while preserving proportions
- Consistent output quality across all chart types

### 5. **Dynamic Date Label Scaling**
Intelligent x-axis label scaling based on data range:

```python
# Adaptive tick calculation
date_range = last_date - first_date
num_ticks = min(8, len(df)) if date_range.days <= 30 else 8 if date_range.days <= 90 else 10

# Dynamic date format
date_format = '%Y-%m-%d' if date_range.days > 30 else '%m-%d'
```

### 6. **Chart Title Positioning & Spacing Control**
The app implements strategic title positioning that places titles **outside the chart body** with controlled spacing:

```python
# Main chart title with controlled padding - positioned OUTSIDE chart area
ax1.set_title(f"{ticker} - Price with EMAs and Bollinger Bands ({frequency})", 
              fontsize=14, fontweight='bold', pad=10, loc='center')

# Subplot titles with consistent styling - positioned ABOVE each subplot
ax2.set_title(f'MACD (12,26,9) - {frequency}', 
              fontsize=12, fontweight='bold', loc='center')

ax3.set_title(f'RSI & ROC - {frequency}', 
              fontsize=12, fontweight='bold', loc='center')

# Combined chart titles with extra padding for clarity
ax1.set_title(f'Daily Chart ({daily_start_str} to {daily_end_str})', 
              fontsize=14, fontweight='bold', pad=10)
ax2.set_title(f'Weekly Chart ({weekly_start_str} to {weekly_end_str})', 
              fontsize=14, fontweight='bold', pad=10)
```

**Key Title Positioning Parameters:**

1. **`pad=10`** - Creates 10 points of spacing between the title and the chart border
   - Ensures titles don't overlap with chart content
   - Provides clean visual separation
   - Consistent across all main charts

2. **`loc='center'`** - Centers titles horizontally above each subplot
   - Creates balanced, professional appearance
   - Aligns with chart content below

3. **Font Scaling Hierarchy**:
   - Main titles: `fontsize=14` for primary emphasis
   - Subplot titles: `fontsize=12` for secondary hierarchy
   - All titles: `fontweight='bold'` for readability

**Key Benefits:**
- **Titles stay outside chart body** - Never interfere with data visualization
- **Consistent spacing** - `pad=10` ensures uniform appearance across all charts
- **Professional hierarchy** - Font sizes create clear visual structure
- **Clean separation** - Padding prevents title-data overlap
- **Responsive positioning** - Titles adapt to different chart sizes while maintaining proportions

**Title Positioning Best Practices from the Code:**
```python
# For main charts - use padding for extra space
ax.set_title("Chart Title", fontsize=14, fontweight='bold', pad=10, loc='center')

# For subplots - standard positioning without extra padding
ax.set_title("Subplot Title", fontsize=12, fontweight='bold', loc='center')

# For combined layouts - consistent padding across both sides
ax1.set_title("Left Chart", fontsize=14, fontweight='bold', pad=10)
ax2.set_title("Right Chart", fontsize=14, fontweight='bold', pad=10)
```

## Complete Code Patches for Implementation

### Patch 1: Basic Chart Creation with Scaling
```python
import matplotlib.pyplot as plt
import tempfile
import os
from datetime import datetime
import pandas as pd

def create_scaled_chart(df, ticker, title, frequency="Daily"):
    """
    Create a proportionally scaled matplotlib chart with proper scaling mechanisms.
    
    Args:
        df: DataFrame with DATE, CLOSE, VOLUME columns
        ticker: Stock symbol
        title: Chart title
        frequency: "Daily" or "Weekly"
    
    Returns:
        str: Path to saved chart image
    """
    # SCALING MECHANISM 1: Strategic figure sizing
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), 
                                       height_ratios=[2, 1, 1], 
                                       sharex=True, 
                                       gridspec_kw={'hspace': 0})
    
    # Create twin axis for volume scaling
    ax1v = ax1.twinx()
    
    # SCALING MECHANISM 2: Volume normalization
    # Calculate proportional bar width
    bar_width = (df['DATE'].iloc[-1] - df['DATE'].iloc[0]).days / len(df) * 0.8
    if bar_width <= 0:
        bar_width = 0.8
        
    # Normalize volume to price range
    price_range = df['CLOSE'].max() - df['CLOSE'].min()
    volume_scale_factor = price_range * 0.2 / df['VOLUME'].max() if df['VOLUME'].max() > 0 else 0.2
    normalized_volume = df['VOLUME'] * volume_scale_factor
    
    # Plot price data
    ax1.plot(df['DATE'], df['CLOSE'], label='Close Price', color='black', linewidth=1.5)
    
    # Plot scaled volume bars
    df['price_change'] = df['CLOSE'].diff()
    volume_colors = ['#26A69A' if val >= 0 else '#EF5350' for val in df['price_change']]
    ax1v.bar(df['DATE'], normalized_volume, width=bar_width, color=volume_colors, alpha=0.3)
    
    # SCALING MECHANISM 3: Volume axis scaling
    ax1v.set_ylabel('Volume', fontsize=10, color='gray')
    ax1v.set_yticklabels([])  # Remove volume labels to avoid clutter
    ax1v.tick_params(axis='y', length=0)
    ax1v.set_ylim(0, price_range * 0.3)  # Limit volume to 30% of price range
    
    # Style main price chart
    ax1.set_title(f"{ticker} - {title} ({frequency})", fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.set_xticklabels([])
    
    # SCALING MECHANISM 4: Dynamic date scaling
    first_date = df['DATE'].iloc[0]
    last_date = df['DATE'].iloc[-1]
    date_range = last_date - first_date
    num_ticks = min(8, len(df)) if date_range.days <= 30 else 8 if date_range.days <= 90 else 10
    
    # Calculate tick positions
    tick_indices = [0] + list(range(len(df) // (num_ticks - 2), len(df) - 1, len(df) // (num_ticks - 2)))[:num_ticks-2] + [len(df) - 1]
    tick_indices = [i for i in tick_indices if 0 <= i < len(df)]
    
    if tick_indices:
        ax3.set_xticks([df['DATE'].iloc[i] for i in tick_indices])
        date_format = '%Y-%m-%d' if date_range.days > 30 else '%m-%d'
        tick_labels = [df['DATE'].iloc[i].strftime(date_format) for i in tick_indices]
        ax3.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Apply tight layout for optimal spacing
    plt.tight_layout()
    
    # SCALING MECHANISM 5: High-quality export with consistent DPI
    temp_dir = tempfile.gettempdir()
    chart_filename = f"{ticker}_{frequency.lower()}_scaled_chart.png"
    temp_path = os.path.join(temp_dir, chart_filename)
    fig.savefig(temp_path, dpi=150, bbox_inches='tight')
    
    plt.close(fig)
    return temp_path
```

### Patch 2: Chart Combination with Proportional Scaling
```python
def combine_charts_with_scaling(chart1_path, chart2_path, title1="Chart 1", title2="Chart 2"):
    """
    Combine two charts side-by-side while maintaining proportional scaling.
    
    Args:
        chart1_path: Path to first chart image
        chart2_path: Path to second chart image
        title1: Title for first chart
        title2: Title for second chart
    
    Returns:
        str: Path to combined chart image
    """
    # Read existing chart images
    img1 = plt.imread(chart1_path)
    img2 = plt.imread(chart2_path)
    
    # SCALING MECHANISM: Use 2:1 aspect ratio for side-by-side layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # Display images without distortion
    ax1.imshow(img1)
    ax2.imshow(img2)
    
    # Remove axes to maintain clean appearance
    ax1.axis('off')
    ax2.axis('off')
    
    # Add titles with proper spacing
    ax1.set_title(title1, fontsize=14, fontweight='bold', pad=10)
    ax2.set_title(title2, fontsize=14, fontweight='bold', pad=10)
    
    # Apply tight layout for optimal spacing
    plt.tight_layout()
    
    # Save with consistent scaling parameters
    temp_dir = tempfile.gettempdir()
    combined_path = os.path.join(temp_dir, "combined_scaled_chart.png")
    fig.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return combined_path
```

### Patch 3: Adaptive Scaling Function
```python
def calculate_adaptive_scaling(data_length, date_range_days, value_range):
    """
    Calculate adaptive scaling parameters based on data characteristics.
    
    Args:
        data_length: Number of data points
        date_range_days: Number of days in date range
        value_range: Range of values (max - min)
    
    Returns:
        dict: Scaling parameters
    """
    # Calculate appropriate figure size based on data density
    if data_length < 50:
        base_width = 10
    elif data_length < 200:
        base_width = 12
    else:
        base_width = 14
    
    # Calculate bar width for optimal spacing
    bar_width = min(0.8, (date_range_days / data_length) * 0.8)
    
    # Calculate volume scaling factor
    volume_scale = value_range * 0.2 if value_range > 0 else 0.2
    
    # Determine tick density
    if date_range_days <= 30:
        num_ticks = min(8, data_length)
        date_format = '%m-%d'
    elif date_range_days <= 90:
        num_ticks = 8
        date_format = '%m-%d'
    else:
        num_ticks = 10
        date_format = '%Y-%m-%d'
    
    return {
        'figsize': (base_width, base_width),
        'bar_width': bar_width,
        'volume_scale': volume_scale,
        'num_ticks': num_ticks,
        'date_format': date_format,
        'dpi': 150
    }
```

### Patch 4: Complete Implementation Example
```python
def implement_chart_scaling_system(df, ticker="STOCK", chart_type="technical"):
    """
    Complete implementation of the chart scaling system from the FastAPI app.
    
    Args:
        df: DataFrame with required columns (DATE, CLOSE, VOLUME, etc.)
        ticker: Stock ticker symbol
        chart_type: Type of chart to create
    
    Returns:
        str: Path to the generated chart
    """
    # Calculate scaling parameters
    date_range = (df['DATE'].iloc[-1] - df['DATE'].iloc[0]).days
    price_range = df['CLOSE'].max() - df['CLOSE'].min()
    scaling_params = calculate_adaptive_scaling(len(df), date_range, price_range)
    
    # Create figure with calculated scaling
    fig, axes = plt.subplots(3, 1, figsize=scaling_params['figsize'], 
                            height_ratios=[2, 1, 1], 
                            sharex=True, 
                            gridspec_kw={'hspace': 0})
    
    ax1, ax2, ax3 = axes
    ax1v = ax1.twinx()
    
    # Apply volume scaling
    volume_scale_factor = scaling_params['volume_scale'] / df['VOLUME'].max() if df['VOLUME'].max() > 0 else 0.2
    normalized_volume = df['VOLUME'] * volume_scale_factor
    
    # Plot with proper scaling
    ax1.plot(df['DATE'], df['CLOSE'], label='Close Price', color='black', linewidth=1.5)
    
    # Volume bars with calculated width
    df['price_change'] = df['CLOSE'].diff()
    volume_colors = ['#26A69A' if val >= 0 else '#EF5350' for val in df['price_change']]
    ax1v.bar(df['DATE'], normalized_volume, width=scaling_params['bar_width'], 
             color=volume_colors, alpha=0.3)
    
    # Apply volume axis scaling
    ax1v.set_ylim(0, price_range * 0.3)
    ax1v.set_ylabel('Volume', fontsize=10, color='gray')
    ax1v.set_yticklabels([])
    ax1v.tick_params(axis='y', length=0)
    
    # Style charts
    ax1.set_title(f"{ticker} - Scaled Technical Chart", fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.2)
    
    # Apply date scaling
    tick_indices = list(range(0, len(df), max(1, len(df) // scaling_params['num_ticks'])))
    if tick_indices[-1] != len(df) - 1:
        tick_indices.append(len(df) - 1)
    
    ax3.set_xticks([df['DATE'].iloc[i] for i in tick_indices])
    tick_labels = [df['DATE'].iloc[i].strftime(scaling_params['date_format']) for i in tick_indices]
    ax3.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save with optimal settings
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"{ticker}_scaled_chart.png")
    fig.savefig(output_path, dpi=scaling_params['dpi'], bbox_inches='tight')
    plt.close(fig)
    
    return output_path
```

## Key Implementation Notes

1. **Always use `bbox_inches='tight'`** - This removes excess whitespace while preserving proportions
2. **DPI=150 is optimal** - Provides high quality without excessive file sizes
3. **Volume scaling is critical** - Normalize volume to 20% of price range, limit axis to 30%
4. **Bar width calculation** - Base on date range and data density for proper proportions
5. **Height ratios [2,1,1]** - Price chart gets 50% of space, indicators get 25% each
6. **Dynamic date formatting** - Adapt label density and format based on time range
7. **Use `plt.close(fig)`** - Always close figures to prevent memory leaks

## Summary

The FastAPI Technical Analysis app implements a sophisticated multi-layered scaling system that:
- Maintains proportional scaling across different data ranges
- Adapts to varying data densities
- Ensures visual consistency and readability
- Produces high-quality, professional charts
- Handles edge cases gracefully

This scaling system can be directly applied to any matplotlib-based charting application to achieve similar professional results with proper proportional scaling.
