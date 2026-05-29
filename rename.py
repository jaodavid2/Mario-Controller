import os
import glob

# Sæt det grundlæggende navn du vil have (f.eks. salgsdata_)
basis_navn = "Hop_"

# Find alle .csv filer i den nuværende mappe
csv_filer = glob.glob("*.csv")

# Sorter dem, så de omdøbes i alfabetiskkronologisk rækkefølge
csv_filer.sort()

# Gå igennem alle filerne og giv dem et nummer (f.eks. salgsdata_1.csv)
for i, filnavn in enumerate(csv_filer, start=1):
    nyt_navn = f"{basis_navn}{i}.csv"
    
    # Omdøb filen
    os.rename(filnavn, nyt_navn)
    print(f"Omdøbte {filnavn} - {nyt_navn}")

print("Færdig! Alle filer er omdøbt.")