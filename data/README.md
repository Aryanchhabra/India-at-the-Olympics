# Olympic Dataset Documentation

This folder contains the Olympic history datasets used for India's medal analysis and prediction.

## 📁 Files

### 1. athlete_events.csv
**Main Olympics Dataset**

- **Size**: ~35 MB
- **Records**: 271,116 athlete entries
- **Time Period**: 1896-2016
- **Columns**: 15

#### Column Description:
| Column | Type | Description |
|--------|------|-------------|
| ID | Integer | Unique athlete ID |
| Name | String | Athlete's name |
| Sex | String | M (Male) or F (Female) |
| Age | Float | Athlete's age (may contain NaN) |
| Height | Float | Height in cm (may contain NaN) |
| Weight | Float | Weight in kg (may contain NaN) |
| Team | String | Team/country name |
| NOC | String | National Olympic Committee 3-letter code |
| Games | String | Year and season (e.g., "2016 Summer") |
| Year | Integer | Olympic year |
| Season | String | Summer or Winter |
| City | String | Host city |
| Sport | String | Sport category |
| Event | String | Specific event |
| Medal | String | Gold, Silver, Bronze, or NaN (no medal) |

#### Key Statistics:
- **Countries**: 230+ NOC codes
- **Sports**: 60+ different sports
- **Events**: 700+ individual events
- **Athletes**: 130K+ unique athletes
- **Olympics**: 51 Olympic games (35 Summer, 16 Winter)

### 2. noc_regions.csv
**NOC Code Mappings**

- **Size**: ~5 KB
- **Records**: 230 NOC regions
- **Columns**: 3

#### Column Description:
| Column | Type | Description |
|--------|------|-------------|
| NOC | String | 3-letter National Olympic Committee code |
| region | String | Country/region name |
| notes | String | Additional notes (often empty) |

#### India's Entry:
- **NOC**: IND
- **Region**: India
- **Notes**: (empty)

## 🇮🇳 India-Specific Data

### Filtering Criteria Used:
- **Team**: "India"
- **Season**: "Summer" only
- **Year**: 1948 onwards (post-independence)

### Expected India Data:
- **Athletes**: 779 unique Indian athletes (1948-2016)
- **Olympic Games**: 18 Summer Olympics
- **Sports**: 23 different sports
- **Actual Unique Medals**: **23** (1948-2016)
  - Gold: 6, Silver: 5, Bronze: 12
- **Medal Records**: 140 (counts each athlete on medal-winning teams)

⚠️ **Important**: Don't confuse "medal records" (140) with "actual medals" (23).
Team sports like Hockey inflate the record count since each player gets an entry.

### Top Sports for India:
1. **Hockey** - Most successful historically
2. **Wrestling** - Strong recent performance
3. **Shooting** - Consistent medals
4. **Boxing** - Emerging strength
5. **Badminton** - Recent success

## 📊 Data Quality Notes

### Missing Values:
- **Age**: ~10% missing
- **Height**: ~60% missing
- **Weight**: ~62% missing
- **Medal**: ~92% missing (most athletes don't win medals)

### Handling Strategy:
- Age, Height, Weight: Filled with median values
- Medal: NaN treated as "No Medal"

## 🔍 Data Source

**Original Source**: Kaggle  
**Dataset**: 120 Years of Olympic History: Athletes and Results  
**Author**: rgriffin (Kaggle username: heesoo37)  
**Link**: https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results

**Original Data Compiled From**:
- www.sports-reference.com (Olympic statistics)
- International Olympic Committee records

## 📝 Usage Notes

### In Jupyter Notebook:
```python
# Load main dataset
df = pd.read_csv('../data/athlete_events.csv')

# Load NOC mappings
noc_df = pd.read_csv('../data/noc_regions.csv')

# Filter for India (Summer Olympics, 1948+)
india_df = df[
    (df['Team'] == 'India') & 
    (df['Season'] == 'Summer') & 
    (df['Year'] >= 1948)
].copy()
```

### In Streamlit App:
The app loads pre-processed CSV files from the `app/` folder:
- `india_olympics_data.csv` - Filtered and cleaned India data
- `yearly_aggregated_data.csv` - Year-level features
- `top_athletes.csv` - Top performing athletes
- `sports_medals.csv` - Sport-wise medal counts

## ⚠️ Important Notes

1. **Data Version**: This dataset goes up to 2016 Olympics. For 2020/2024 data, you'll need to source separately.

2. **Team Names**: Some historical variations exist (e.g., "India" vs "British India"). Our filter uses "India" for post-1947 data.

3. **Medal Counts**: In team sports, each team member gets counted individually, so medal counts may seem higher than official tallies.

4. **Data Accuracy**: While comprehensive, there may be minor discrepancies with official IOC records.

## 🔄 Updates

To update with newer Olympic data:
1. Download updated dataset from Kaggle or Olympic databases
2. Replace the CSV files in this folder
3. Re-run the Jupyter notebook to regenerate processed files
4. Restart the Streamlit app

## 📄 License

The dataset is provided under the Database Contents License (DbCL) by the original author on Kaggle. Please refer to the Kaggle page for detailed licensing information.

---

**Last Verified**: October 2025  
**Dataset Version**: Up to 2016 Rio Olympics

