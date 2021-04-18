## README.txt
#example run 
python3 ./determineCarPrices.py ./vehicles_medium.csv -v -hu "[20, 20]" -out "[price]" -in "[year, manufacturer, model, condition, cylinders, fuel, odometer, title_status, transmission, drive, paint_color]" -u "[2010, 1, 1, 1, 1, 1, 43000, 1, 1, 1, 1]"
python3 ./determineCarPrices.py ./vehicles_medium.csv -out "[price]" -in "[year, manufacturer, model, condition, cylinders, fuel, odometer, title_status, transmission, drive, paint_color]" -u "[2014, 2, 2, 2, 2, 1, 93000, 1, 2, 2, 0]"
python3 ./determineCarPrices.py ./vehicles_small.csv -v -hu "[20, 20]" -out "[price]" -in "[year]"

## New changes
python3 ./determineCarPrices.py -train_csv ./vehicles_large.csv -hu "[20, 20]" -out "[price]" -in "[year, manufacturer, model, condition, cylinders, fuel, odometer, title_status, transmission, drive, paint_color]"
python3 ./determineCarPrices.py -u "[2010, 1, 1, 1, 1, 1, 43000, 1, 1, 1, 1]"


##Distributed NN RUN
#if the processedData.csv is not copied first run
python3 ./cleanAndSaveData.py -train_csv ./vehicles_large.csv -hu "[20, 20]" -out "[price]" -in "[year, manufacturer, model, condition, cylinders, fuel, odometer, title_status, transmission, drive, paint_color]"

#Otherwise only run (for four machine ws = 4 and r 0 to 3) 
#in master machine
python3 ./distDetermineCarPrice.py -r 0 -ws 4

# in second machine 
python3 ./distDetermineCarPrice.py -r 1 -ws 4

# in 3rd machine 
python3 ./distDetermineCarPrice.py -r 2 -ws 4
# in 4th machine 
python3 ./distDetermineCarPrice.py -r 3 -ws 4
