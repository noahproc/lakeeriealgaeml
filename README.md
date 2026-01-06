# Lake Erie Algae Bloom - Simple, Supervised ML
On the top level, my goal was to predict the chlorphyll-A levels in the western basin of Lake erie, which are a key predictor of algae in the Lake based on turbidity, temperature, and date. The western basin experiences the most significant HAB (harmful algae bloom) threat, and in 2014, the algae levels in the lake contaminated water in Toledo, OH for three days. Monitoring the lake's algae levels and maintaining the proper filtration/distribution infrastructure is vital for both the safety and security of the citizens. 

# __Datasets__:

__Weekly Sampling Data__: 
https://www.glerl.noaa.gov/res/HABs_and_Hypoxia/wle-weekly-current
NOAA (National Oceanic and Atmospheric Administration) released weekly enviornmental datasets from buoy stations in the western basin. This data is what was used to train and test the model, as well as to plot the stations on the shoreline map.

__Shoreline Data__: https://shoreline.noaa.gov/med-res.html
Also from the NOAA, this is a shp file that maps the Lake Erie shoreline. I used GeoPandas to create a dataframe of the shoreline, and then plotted it with Matplotlib.

# __Files__:

| Filename | Function |
| alg_graphical.ipynb | All modeling is done in this file (graphical, SciKit nueral networds, application). |
| user_alg.py | Allows users to provide a temperature or turbidity value and returns the projected chlorophyll-A level. |
| lake_erie_stations.png | Plot of the different station locations. |
| 2025_WLE_Weekly_Datashare_CSV.csv | CSV of the NOAA weekly sampling data. |

All of the .sav models are those saved for export for use in user_alg.py. More information is provided in the notebook. 
