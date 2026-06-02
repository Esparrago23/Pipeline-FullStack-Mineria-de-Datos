mkdir data\raw

Invoke-WebRequest -Uri "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select%20*%20from%20cumulative&format=csv" -OutFile "mineria\data\kepler_koi_cumulative.csv"

Invoke-WebRequest -Uri "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select%20*%20from%20pscomppars&format=csv" -OutFile "mineria\data\nasa_pscomppars.csv"